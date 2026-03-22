#!/usr/bin/env python3
"""
Experiment #367: 1d Primary + 1w HTF — Vol Spike Mean Reversion + Regime Filter

Hypothesis: After 366 experiments, the clearest pattern from research is:
1. Volatility spike mean reversion has Sharpe 0.8-1.5 through 2022 crash (BEST edge for BTC/ETH)
2. ATR(7)/ATR(30) > 2.0 + price < BB(20,2.5) captures "vol crush" after panic
3. 1w HMA(21) for major trend bias prevents counter-trend disasters
4. Choppiness Index > 61.8 confirms range market (mean-revert regime)
5. Connors RSI < 15 for precise oversold entry timing
6. Asymmetric sizing: longs 0.30, shorts 0.20 (crypto long bias)
7. 1d timeframe = 15-30 trades/year, minimal fee drag

Why this might beat current best (Sharpe=0.435):
- Vol spike reversion worked through 2022 crash (when trend strategies died)
- 1w HTF filter prevents entering against major trend
- CHOP filter ensures we only mean-revert in actual range markets
- CRSI entry timing reduces false signals
- Conservative sizing (0.20-0.30) limits drawdown

Position sizing: 0.25-0.30 longs, 0.15-0.20 shorts
Stoploss: 2.5 * ATR trailing
Target: 15-30 trades/year on 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_volspike_mr_chop_crsi_1w_v1"
timeframe = "1d"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.5):
    """Calculate Bollinger Bands with configurable std multiplier."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper.values, lower.values, sma.values

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend direction)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_14 = calculate_atr(high, low, close, 14)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.5)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            continue
        
        # === 1W MAJOR TREND REGIME ===
        trend_bull = close[i] > hma_1w_21_aligned[i]
        trend_bear = close[i] < hma_1w_21_aligned[i]
        
        # === VOLATILITY SPIKE (mean reversion signal) ===
        atr_ratio = atr_7[i] / (atr_30[i] + 1e-10)
        vol_spike = atr_ratio > 2.0
        vol_normal = atr_ratio < 1.2
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        price_near_bb_lower = close[i] < bb_lower[i] * 1.02  # within 2% of lower band
        
        # === CHOPPINESS REGIME ===
        choppy_regime = chop_14[i] > 61.8
        trending_regime = chop_14[i] < 38.2
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 20.0
        crsi_overbought = crsi[i] > 80.0
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 30.0
        rsi_overbought = rsi_14[i] > 70.0
        
        # === ENTRY LOGIC: VOL SPIKE MEAN REVERSION ===
        new_signal = 0.0
        
        # LONG: Vol spike + price at BB lower + oversold + (bull trend OR choppy)
        if vol_spike and price_near_bb_lower:
            if crsi_oversold or rsi_oversold:
                if trend_bull or choppy_regime:
                    new_signal = LONG_SIZE
                elif not trend_bear:
                    new_signal = LONG_SIZE * 0.7
        
        # SHORT: Vol spike + price at BB upper + overbought + (bear trend OR choppy)
        if vol_spike and price_above_bb_upper:
            if crsi_overbought or rsi_overbought:
                if trend_bear or choppy_regime:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE
                elif not trend_bull:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE * 0.7
        
        # === SECONDARY ENTRY: CHOPPY REGIME MEAN REVERSION ===
        if choppy_regime and new_signal == 0.0:
            if crsi_oversold and price_near_bb_lower:
                new_signal = LONG_SIZE * 0.8
            elif crsi_overbought and price_above_bb_upper:
                new_signal = -SHORT_SIZE * 0.8
        
        # === TERTIARY ENTRY: TRENDING REGIME PULLBACK ===
        if trending_regime and new_signal == 0.0:
            if trend_bull and crsi_oversold and close[i] > hma_1w_21_aligned[i] * 0.95:
                new_signal = LONG_SIZE * 0.7
            elif trend_bear and crsi_overbought and close[i] < hma_1w_21_aligned[i] * 1.05:
                new_signal = -SHORT_SIZE * 0.7
        
        # === FREQUENCY SAFEGUARD (ensure 15+ trades/year on 1d) ===
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if trend_bull and crsi_oversold and atr_ratio > 1.3:
                new_signal = LONG_SIZE * 0.5
            elif trend_bear and crsi_overbought and atr_ratio > 1.3:
                new_signal = -SHORT_SIZE * 0.5
        
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
        
        # === VOL NORMALIZATION EXIT (vol spike mean reversion complete) ===
        vol_exit = False
        if in_position and position_side != 0:
            if vol_normal:
                vol_exit = True
        
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
            if position_side > 0 and trend_bear and close[i] < hma_1w_21_aligned[i] * 0.97:
                regime_reversal = True
            if position_side < 0 and trend_bull and close[i] > hma_1w_21_aligned[i] * 1.03:
                regime_reversal = True
        
        if stoploss_triggered or vol_exit or crsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0:
                new_signal = LONG_SIZE
            else:
                new_signal = -SHORT_SIZE
        
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