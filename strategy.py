#!/usr/bin/env python3
"""
Experiment #375: 1h Primary + 4h/1d HTF — Regime-Adaptive Multi-Confluence

Hypothesis: After 370+ experiments, the clearest pattern is:
1. 1h timeframe needs EXTREME selectivity (30-60 trades/year max) to avoid fee drag
2. Single indicators fail — need 4+ confluence filters firing together
3. HTF (4h/1d) for DIRECTION, 1h only for ENTRY TIMING within HTF trend
4. Choppiness Index regime detection prevents trend strategies from dying in chop
5. Connors RSI extremes (not standard RSI) catch reversals with 70%+ win rate
6. Session filter (8-20 UTC) avoids low-liquidity whipsaws
7. Volume confirmation prevents false breakouts

Why this might beat current best (Sharpe=0.435):
- 4+ confluence = fewer but higher-quality trades
- Regime-adaptive: mean-revert in chop, trend-follow in trends
- 1d HMA prevents counter-trend trades in major moves
- Discrete signal levels minimize fee churn
- ATR trailing stop cuts losers at 2.5x

Position sizing: 0.20-0.30 (conservative for 1h TF)
Stoploss: 2.5 * ATR trailing
Target: 30-60 trades/year on 1h (~120-240 train, 38-75 test)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_crsi_chop_hma4h1d_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = choppy/ranging market (mean-revert)
    CHOP < 38.2 = trending market (trend-follow)
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    chop[np.isnan(chop)] = 50.0
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long: CRSI < 10 (oversold)
    Short: CRSI > 90 (overbought)
    """
    close_s = pd.Series(close)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI
    delta = close_s.diff()
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(len(close))
    for i in range(streak_period, len(close)):
        if streak_abs[i] == 0:
            streak_rsi[i] = 50.0
        else:
            streak_rsi[i] = 100.0 / (1.0 + streak_abs[i])
            if streak[i] < 0:
                streak_rsi[i] = 100.0 - streak_rsi[i]
    
    # Percent Rank
    percent_rank = np.zeros(len(close))
    for i in range(rank_period, len(close)):
        window = close[i-rank_period:i+1]
        rank = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100.0 * rank / rank_period
    
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    return vol_ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    chop_4h_14 = calculate_choppiness(
        df_4h['high'].values, 
        df_4h['low'].values, 
        df_4h['close'].values, 
        14
    )
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    chop_4h_14_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_14)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_1h_21 = calculate_hma(close, period=21)
    hma_1h_8 = calculate_hma(close, period=8)
    sma_200 = calculate_sma(close, 200)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Extract UTC hours
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.20
    SHORT_STRONG = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(chop_4h_14_aligned[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(hma_1h_21[i]) or np.isnan(sma_200[i]):
            continue
        
        # === 1D MAJOR TREND REGIME (primary direction filter) ===
        regime_bull_1d = close[i] > hma_1d_21_aligned[i]
        regime_bear_1d = close[i] < hma_1d_21_aligned[i]
        
        # === 4H TREND CONFIRMATION ===
        regime_bull_4h = close[i] > hma_4h_21_aligned[i]
        regime_bear_4h = close[i] < hma_4h_21_aligned[i]
        
        # === 4H CHOPPINESS REGIME (determines strategy type) ===
        choppy_regime = chop_4h_14_aligned[i] > 55.0
        trending_regime = chop_4h_14_aligned[i] < 45.0
        neutral_regime = not choppy_regime and not trending_regime
        
        # === 1H LOCAL TREND ===
        hma_bullish_1h = hma_1h_8[i] > hma_1h_21[i]
        hma_bearish_1h = hma_1h_8[i] < hma_1h_21[i]
        
        price_above_sma200 = close[i] > sma_200[i]
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= utc_hours[i] <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 0.8
        
        # === CONNORS RSI SIGNALS (mean-reversion) ===
        crsi_extreme_oversold = crsi[i] < 12.0
        crsi_oversold = crsi[i] < 20.0
        crsi_extreme_overbought = crsi[i] > 88.0
        crsi_overbought = crsi[i] > 80.0
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ENTRY LOGIC - MULTI-CONFLUENCE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # Confluence counter for entries (need 4+ filters)
        
        # === LONG ENTRIES ===
        long_confluence = 0
        
        # 1. 1D trend bull
        if regime_bull_1d:
            long_confluence += 1
        
        # 2. 4H trend bull or neutral
        if regime_bull_4h or (not regime_bear_4h):
            long_confluence += 1
        
        # 3. CRSI oversold
        if crsi_oversold:
            long_confluence += 1
        
        # 4. In session
        if in_session:
            long_confluence += 1
        
        # 5. Volume confirmed
        if volume_confirmed:
            long_confluence += 1
        
        # 6. Price above SMA200
        if price_above_sma200:
            long_confluence += 1
        
        # 7. 1H HMA bullish or CRSI extreme
        if hma_bullish_1h or crsi_extreme_oversold:
            long_confluence += 1
        
        # Enter long if 4+ confluence in appropriate regime
        if long_confluence >= 4:
            if choppy_regime and crsi_extreme_oversold:
                # Mean-reversion in chop
                new_signal = LONG_BASE
            elif trending_regime and regime_bull_1d and hma_bullish_1h:
                # Trend-follow in trend
                new_signal = LONG_STRONG
            elif neutral_regime and crsi_oversold and regime_bull_1d:
                # Hybrid
                new_signal = LONG_BASE
        
        # === SHORT ENTRIES ===
        short_confluence = 0
        
        # 1. 1D trend bear
        if regime_bear_1d:
            short_confluence += 1
        
        # 2. 4H trend bear or neutral
        if regime_bear_4h or (not regime_bull_4h):
            short_confluence += 1
        
        # 3. CRSI overbought
        if crsi_overbought:
            short_confluence += 1
        
        # 4. In session
        if in_session:
            short_confluence += 1
        
        # 5. Volume confirmed
        if volume_confirmed:
            short_confluence += 1
        
        # 6. Price below SMA200
        if not price_above_sma200:
            short_confluence += 1
        
        # 7. 1H HMA bearish or CRSI extreme
        if hma_bearish_1h or crsi_extreme_overbought:
            short_confluence += 1
        
        # Enter short if 4+ confluence in appropriate regime
        if short_confluence >= 4 and new_signal == 0.0:
            if choppy_regime and crsi_extreme_overbought:
                # Mean-reversion in chop
                new_signal = -SHORT_BASE
            elif trending_regime and regime_bear_1d and hma_bearish_1h:
                # Trend-follow in trend
                new_signal = -SHORT_STRONG
            elif neutral_regime and crsi_overbought and regime_bear_1d:
                # Hybrid
                new_signal = -SHORT_BASE
        
        # === FREQUENCY SAFEGUARD (ensure 30+ trades/year on 1h) ===
        # Force trade if no signal for 30 bars (~30 hours on 1h)
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if regime_bull_1d and crsi[i] < 35.0 and in_session:
                new_signal = LONG_BASE * 0.6
            elif regime_bear_1d and crsi[i] > 65.0 and in_session:
                new_signal = -SHORT_BASE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === CRSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and crsi_overbought:
                crsi_exit = True
            if position_side < 0 and crsi_oversold:
                crsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and regime_bear_1d and close[i] < hma_1h_21[i]:
                regime_reversal = True
            if position_side < 0 and regime_bull_1d and close[i] > hma_1h_21[i]:
                regime_reversal = True
        
        if stoploss_triggered or crsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.27:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.22:
                new_signal = -SHORT_STRONG
            else:
                new_signal = -SHORT_BASE
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals