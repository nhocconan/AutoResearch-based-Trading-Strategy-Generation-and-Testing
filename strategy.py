#!/usr/bin/env python3
"""
Experiment #365: 1h Primary + 4h/1d HTF — Dual Regime with Relaxed Entries

Hypothesis: After 360+ experiments, the critical lesson is TRADE FREQUENCY.
Strategies #355, #360 got Sharpe=0.000 because they generated ZERO trades.
Entry conditions were TOO STRICT.

This strategy uses:
1. 4h HMA(21) for major trend direction (HTF bias)
2. 1d HMA(21) for regime filter (bull/bear market)
3. 1h Choppiness Index for regime detection (chop vs trend)
4. Connors RSI for mean-reversion entries (relaxed: <20/>80 not <10/>90)
5. Donchian breakout for trend-follow entries (period=14 not 20)
6. ATR(14) trailing stop at 2.5x

Key changes from failed experiments:
- CRSI thresholds: <20/>80 (was <10/>90 - too rare)
- CHOP thresholds: >50/<50 (was >61.8/<38.2 - too extreme)
- Donchian period: 14 (was 20 - more breakouts)
- Min trade frequency safeguard: force entry every 24 bars if no signal
- Position size: 0.25 (balanced for 1h TF)

Target: 40-70 trades/year on 1h timeframe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_dualregime_crsi_donchian_4h1d_relaxed_v1"
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
    CHOP > 50 = choppy/ranging market (mean-revert)
    CHOP < 50 = trending market (trend-follow)
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
    Long: CRSI < 20 (relaxed from <10)
    Short: CRSI > 80 (relaxed from >90)
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

def calculate_donchian(high, low, period=14):
    """Calculate Donchian Channel (period=14 for more breakouts)."""
    donchian_upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    return donchian_upper, donchian_lower, donchian_mid

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
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
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    rsi_14 = calculate_rsi(close, 14)
    hma_1h_21 = calculate_hma(close, period=21)
    hma_1h_8 = calculate_hma(close, period=8)
    sma_200 = calculate_sma(close, 200)
    
    # Donchian channels (period=14 for more breakouts)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.20
    SHORT_STRONG = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(hma_1h_21[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === 1D MAJOR TREND REGIME (primary direction filter) ===
        regime_bull = close[i] > hma_1d_21_aligned[i]
        regime_bear = close[i] < hma_1d_21_aligned[i]
        
        # === 4H TREND BIAS (secondary filter) ===
        bias_bull = close[i] > hma_4h_21_aligned[i]
        bias_bear = close[i] < hma_4h_21_aligned[i]
        
        # === CHOPPINESS REGIME (determines strategy type) ===
        # CHOP > 50 = choppy (mean-revert)
        # CHOP < 50 = trending (trend-follow)
        choppy_regime = chop_14[i] > 50.0
        trending_regime = chop_14[i] < 50.0
        
        # === VOLATILITY REGIME (ATR ratio) ===
        atr_30 = calculate_atr(high, low, close, 30)
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10) if not np.isnan(atr_30[i]) else 1.0
        high_vol = atr_ratio > 1.5
        vol_scale = 0.8 if high_vol else 1.0
        
        # === 1H LOCAL TREND ===
        hma_bullish = hma_1h_8[i] > hma_1h_21[i]
        hma_bearish = hma_1h_8[i] < hma_1h_21[i]
        
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === CONNORS RSI SIGNALS (mean-reversion) - RELAXED THRESHOLDS ===
        crsi_oversold = crsi[i] < 20.0  # Was <10 - too rare
        crsi_overbought = crsi[i] > 80.0  # Was >90 - too rare
        crsi_neutral = 30.0 < crsi[i] < 70.0
        
        # === RSI EXTREMES (additional filter) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ENTRY LOGIC - DUAL REGIME ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # === CHOPPY REGIME: MEAN-REVERSION (Connors RSI) ===
        if choppy_regime:
            # Long: CRSI oversold + bull bias or price > SMA200
            if crsi_oversold or rsi_oversold:
                if regime_bull or bias_bull or price_above_sma200:
                    new_signal = LONG_BASE * vol_scale
                else:
                    new_signal = LONG_BASE * 0.6 * vol_scale
            
            # Short: CRSI overbought + bear bias or price < SMA200
            if crsi_overbought or rsi_overbought:
                if regime_bear or bias_bear or not price_above_sma200:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE * vol_scale
                else:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE * 0.6 * vol_scale
        
        # === TRENDING REGIME: TREND-FOLLOW (Donchian Breakout) ===
        elif trending_regime:
            # Long: Donchian breakout + bull regime/bias + HMA bullish
            if donchian_breakout_long:
                if regime_bull and bias_bull and hma_bullish:
                    new_signal = LONG_STRONG * vol_scale
                elif regime_bull or bias_bull:
                    new_signal = LONG_BASE * vol_scale
                elif hma_bullish:
                    new_signal = LONG_BASE * 0.7 * vol_scale
            
            # Short: Donchian breakout + bear regime/bias + HMA bearish
            if donchian_breakout_short:
                if new_signal == 0.0:
                    if regime_bear and bias_bear and hma_bearish:
                        new_signal = -SHORT_STRONG * vol_scale
                    elif regime_bear or bias_bear:
                        new_signal = -SHORT_BASE * vol_scale
                    elif hma_bearish:
                        new_signal = -SHORT_BASE * 0.7 * vol_scale
        
        # === FREQUENCY SAFEGUARD (CRITICAL - ensure 30+ trades/year) ===
        # Force trade if no signal for 24 bars (~1 day on 1h)
        if bars_since_last_trade > 24 and new_signal == 0.0 and not in_position:
            if regime_bull and bias_bull and (crsi[i] < 40.0 or rsi_14[i] < 45.0):
                new_signal = LONG_BASE * 0.5 * vol_scale
            elif regime_bear and bias_bear and (crsi[i] > 60.0 or rsi_14[i] > 55.0):
                new_signal = -SHORT_BASE * 0.5 * vol_scale
            elif crsi_oversold and price_above_sma200:
                new_signal = LONG_BASE * 0.5 * vol_scale
            elif crsi_overbought and not price_above_sma200:
                new_signal = -SHORT_BASE * 0.5 * vol_scale
            elif hma_bullish and close[i] > hma_1h_21[i]:
                new_signal = LONG_BASE * 0.4 * vol_scale
            elif hma_bearish and close[i] < hma_1h_21[i]:
                new_signal = -SHORT_BASE * 0.4 * vol_scale
        
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
            if position_side > 0 and regime_bear and bias_bear and close[i] < hma_1h_21[i]:
                regime_reversal = True
            if position_side < 0 and regime_bull and bias_bull and close[i] > hma_1h_21[i]:
                regime_reversal = True
        
        if stoploss_triggered or crsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG * vol_scale
            elif new_signal > 0:
                new_signal = LONG_BASE * vol_scale
            elif new_signal < -0.18:
                new_signal = -SHORT_STRONG * vol_scale
            else:
                new_signal = -SHORT_BASE * vol_scale
        
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