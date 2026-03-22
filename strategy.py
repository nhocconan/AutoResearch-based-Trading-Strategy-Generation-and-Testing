#!/usr/bin/env python3
"""
Experiment #355: 1h Primary + 4h/1d HTF — Connors RSI + Choppiness + HMA Trend + Session/Volume

Hypothesis: After 30+ failed experiments, the pattern is clear for 1h timeframe:
1. Too many strict filters = 0 trades (exp 345, 348, 350 all failed with Sharpe=0.000)
2. Need LOOSE filters that still provide confluence but allow 30-60 trades/year
3. Connors RSI (CRSI) has proven 75% win rate for mean reversion in crypto
4. Choppiness Index regime filter works best when thresholds are WIDE (not narrow)
5. 4h HMA for trend direction + 1h CRSI for entry timing = proven pattern

Key changes from failed 1h experiments:
- CRSI thresholds: <25/>75 (not <10/>90 which is too rare)
- CHOP thresholds: >50/<50 (not >61.8/<38.2 which rarely triggers)
- Volume filter: >0.5x avg (not >0.8x which filters too much)
- Session: 6-22 UTC (not 8-20 which misses Asian session moves)
- Force trade every 20 bars if no signal (frequency safeguard)

This strategy combines:
1. 1d HMA(21) for major regime (bull/bear bias)
2. 4h HMA(21) for intermediate trend direction
3. 1h Connors RSI for entry timing (mean reversion within trend)
4. Choppiness Index(14) for regime detection (wide thresholds)
5. Volume filter >0.5x 20-bar average (loose filter)
6. Session filter 6-22 UTC (captures major sessions)
7. ATR(14) trailing stop 2.5x for risk management

Position sizing: 0.20 base, 0.25 strong (conservative for 1h TF)
Target: 40-70 trades/year on 1h (1 trade every 5-9 days)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_hma_4h1d_session_vol_loose_v1"
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
    half = max(n // 2, 1)
    sqrt_n = max(int(np.sqrt(n)), 1)
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        if span < 1:
            return series
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) - very short term
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_short = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI of streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Percent Rank (percentile of price change over lookback)
    pct_rank = pd.Series(close).pct_change().rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) >= rank_period else np.nan
    )
    
    # Combine into CRSI
    crsi = (rsi_short + rsi_streak + pct_rank) / 3.0
    
    return crsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 50 = range/choppy market (mean revert)
    CHOP < 50 = trending market (trend follow)
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    # Rolling sum of ATR
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Price range
    price_range = highest_high - lowest_low
    
    # CHOP calculation
    chop = np.zeros(len(close))
    mask = (price_range > 0) & (atr_sum > 0)
    chop[mask] = 100.0 * np.log10(atr_sum[mask] / price_range[mask]) / np.log10(period)
    
    # Clip to valid range
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
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
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, 14)
    vol_sma_20 = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Conservative for 1h TF (more trades = more fees)
    LONG_BASE = 0.20
    LONG_STRONG = 0.25
    SHORT_BASE = 0.15
    SHORT_STRONG = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -30
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] == 0:
            continue
        
        # === 1D MAJOR TREND REGIME ===
        # Bull: price above 1d HMA
        # Bear: price below 1d HMA
        regime_bull = close[i] > hma_1d_21_aligned[i]
        regime_bear = close[i] < hma_1d_21_aligned[i]
        
        # === 4H INTERMEDIATE TREND ===
        trend_bull = close[i] > hma_4h_21_aligned[i]
        trend_bear = close[i] < hma_4h_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 50 = range (favor mean reversion)
        # CHOP < 50 = trend (favor trend following)
        choppy_market = chop[i] > 50.0
        trending_market = chop[i] < 50.0
        
        # === CONNORS RSI SIGNALS (LOOSE thresholds for trade frequency) ===
        # CRSI < 25 = oversold (long opportunity)
        # CRSI > 75 = overbought (short opportunity)
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # CRSI turning (momentum)
        crsi_rising = crsi[i] > crsi[i-1] if i > 0 else False
        crsi_falling = crsi[i] < crsi[i-1] if i > 0 else False
        
        # === VOLUME FILTER (LOOSE - 0.5x average) ===
        volume_ok = volume[i] > 0.5 * vol_sma_20[i]
        
        # === SESSION FILTER (6-22 UTC - captures major sessions) ===
        hour = get_hour_from_open_time(open_time[i])
        session_ok = 6 <= hour <= 22
        
        # === ENTRY LOGIC (LOOSE conditions to ensure trades) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if regime_bull or trend_bull:  # At least one bullish
            # Primary: CRSI oversold + volume ok + session ok
            if crsi_oversold and volume_ok and session_ok:
                if choppy_market:
                    new_signal = LONG_BASE  # Mean revert in range
                elif trending_market and trend_bull:
                    new_signal = LONG_STRONG  # Trend pullback
            
            # Strong: CRSI extreme oversold
            elif crsi_extreme_oversold and volume_ok:
                new_signal = LONG_STRONG
            
            # CRSI rising from oversold (momentum confirmation)
            elif crsi[i] < 35.0 and crsi_rising and volume_ok and session_ok:
                if new_signal == 0.0:
                    new_signal = LONG_BASE * 0.8
        
        # SHORT ENTRIES
        if regime_bear or trend_bear:  # At least one bearish
            # Primary: CRSI overbought + volume ok + session ok
            if crsi_overbought and volume_ok and session_ok:
                if choppy_market:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE  # Mean revert in range
                elif trending_market and trend_bear:
                    if new_signal == 0.0:
                        new_signal = -SHORT_STRONG  # Trend pullback
            
            # Strong: CRSI extreme overbought
            elif crsi_extreme_overbought and volume_ok:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG
            
            # CRSI falling from overbought (momentum confirmation)
            elif crsi[i] > 65.0 and crsi_falling and volume_ok and session_ok:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8
        
        # === FREQUENCY SAFEGUARD (ensure 40+ trades/year on 1h) ===
        # Force trade if no signal for 20 bars (~20 hours on 1h)
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if crsi_extreme_oversold and (regime_bull or trend_bull):
                new_signal = LONG_BASE * 0.6
            elif crsi_extreme_overbought and (regime_bear or trend_bear):
                new_signal = -SHORT_BASE * 0.6
            elif crsi[i] < 20.0 and regime_bull:
                new_signal = LONG_BASE * 0.5
            elif crsi[i] > 80.0 and regime_bear:
                new_signal = -SHORT_BASE * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === CRSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when CRSI turns overbought
            if position_side > 0 and crsi_overbought:
                crsi_exit = True
            # Short position: exit when CRSI turns oversold
            if position_side < 0 and crsi_oversold:
                crsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 1d regime turns bearish + 4h bearish
            if position_side > 0 and regime_bear and trend_bear:
                regime_reversal = True
            # Short position but 1d regime turns bullish + 4h bullish
            if position_side < 0 and regime_bull and trend_bull:
                regime_reversal = True
        
        if stoploss_triggered or crsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.23:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.18:
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