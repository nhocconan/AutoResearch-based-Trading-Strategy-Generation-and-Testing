#!/usr/bin/env python3
"""
Experiment #854: 4h Primary + 12h/1d HTF — Adaptive Regime with HMA + RSI + Fisher

Hypothesis: After 590+ failed strategies, the winning pattern combines:
1. 4h primary timeframe (proven: 20-50 trades/year, optimal fee/trade ratio)
2. 12h HMA(21) for trend bias (smooth, lag-reduced vs EMA)
3. 1d Choppiness Index for regime detection (range vs trend)
4. 4h RSI(7) for pullback entries (faster than RSI14, catches more reversals)
5. 4h Fisher Transform(9) for reversal confirmation in ranging markets
6. 4h ATR(14) for trailing stop (2.5x) and volatility scaling
7. ADAPTIVE sizing: 0.30 in trending regime, 0.20 in ranging regime

Why this should work:
- 4h TF balances signal frequency (enough trades) vs noise (not too many)
- HMA reduces lag vs EMA, critical for 2022 crash and 2025 bear market
- Fisher Transform excels at catching reversals in bear market rallies
- Choppiness regime switch prevents trend strategies from dying in ranges
- Relaxed entry thresholds ensure >=10 trades per symbol

Key lessons from failures:
- #844 (4h KAMA+ADX): Too many filters = negative Sharpe
- #849 (4h KAMA+RSI): Missing regime filter = whipsaw losses
- #851 (4h HMA simple): No reversal logic = fails in bear market
- Need BOTH trend-following AND mean-reversion logic

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive
Timeframe: 4h (target 30-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_fisher_chop_regime_rsi_12h1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average - reduces lag significantly."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform — normalizes price to Gaussian distribution.
    Range typically -1.5 to +1.5. Reversals at extremes.
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_prev
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2.0
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        range_val = highest_high - lowest_low
        if range_val < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else 0.0
            fisher_prev[i] = fisher[i-1] if i > 1 and not np.isnan(fisher[i-1]) else 0.0
            continue
        
        normalized = (hl2 - lowest_low) / range_val
        normalized = np.clip(normalized, 0.001, 0.999)
        
        fisher_val = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher[i] = 0.7 * fisher_val + 0.3 * fisher[i-1]
            fisher_prev[i] = fisher[i-1]
        else:
            fisher[i] = fisher_val
            fisher_prev[i] = fisher_val
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di_pct = 100 * plus_di / (atr + 1e-10)
        minus_di_pct = 100 * minus_di / (atr + 1e-10)
        dx = 100 * np.abs(plus_di_pct - minus_di_pct) / (plus_di_pct + minus_di_pct + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=7)  # Faster RSI for 4h
    chop_4h = calculate_choppiness(high, low, close, period=14)
    adx_4h = calculate_adx(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    fisher_4h, fisher_prev_4h = calculate_fisher_transform(high, low, period=9)
    
    # Calculate and align 12h HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d Choppiness for regime confirmation
    chop_1d_raw = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    # Calculate 4h HMA for crossover signals
    hma_4h_fast = calculate_hma(close, 16)
    hma_4h_slow = calculate_hma(close, 48)
    
    signals = np.zeros(n)
    TREND_SIZE = 0.30  # Larger size in trending regime
    RANGE_SIZE = 0.20  # Smaller size in ranging regime
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(chop_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if np.isnan(adx_4h[i]) or np.isnan(fisher_4h[i]) or np.isnan(fisher_prev_4h[i]):
            continue
        if atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            continue
        if np.isnan(hma_4h_fast[i]) or np.isnan(hma_4h_slow[i]):
            continue
        
        # === TREND BIAS (12h HTF HMA21) ===
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # === REGIME DETECTION (4h Choppiness + 1d confirmation) ===
        ranging_regime = chop_4h[i] > 55 or (not np.isnan(chop_1d_aligned[i]) and chop_1d_aligned[i] > 55)
        trending_regime = chop_4h[i] < 45 and (np.isnan(chop_1d_aligned[i]) or chop_1d_aligned[i] < 50)
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx_4h[i] > 25
        weak_trend = adx_4h[i] < 20
        
        # === RSI SIGNALS (7-period for 4h) ===
        rsi_oversold = rsi_4h[i] < 40
        rsi_overbought = rsi_4h[i] > 60
        rsi_extreme_oversold = rsi_4h[i] < 25
        rsi_extreme_overbought = rsi_4h[i] > 75
        rsi_cross_up = rsi_4h[i] > 45 and rsi_4h[i-1] <= 45 if i > 0 and not np.isnan(rsi_4h[i-1]) else False
        rsi_cross_down = rsi_4h[i] < 55 and rsi_4h[i-1] >= 55 if i > 0 and not np.isnan(rsi_4h[i-1]) else False
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher_4h[i] < -1.2
        fisher_overbought = fisher_4h[i] > 1.2
        fisher_cross_up = fisher_prev_4h[i] < -1.0 and fisher_4h[i] >= -1.0
        fisher_cross_down = fisher_prev_4h[i] > 1.0 and fisher_4h[i] <= 1.0
        fisher_recovering = fisher_4h[i] > fisher_prev_4h[i] and fisher_4h[i] < -0.5
        fisher_weakening = fisher_4h[i] < fisher_prev_4h[i] and fisher_4h[i] > 0.5
        
        # === HMA CROSSOVER ===
        hma_bullish = hma_4h_fast[i] > hma_4h_slow[i]
        hma_bearish = hma_4h_fast[i] < hma_4h_slow[i]
        hma_cross_up = hma_4h_fast[i] > hma_4h_slow[i] and hma_4h_fast[i-1] <= hma_4h_slow[i-1] if i > 0 else False
        hma_cross_down = hma_4h_fast[i] < hma_4h_slow[i] and hma_4h_fast[i-1] >= hma_4h_slow[i-1] if i > 0 else False
        
        desired_signal = 0.0
        current_size = TREND_SIZE if trending_regime else RANGE_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 45) ===
        if trending_regime and strong_trend:
            # Long: Bullish trend + HMA cross or RSI pullback
            if trend_12h_bullish or hma_bullish:
                if hma_cross_up and rsi_oversold:
                    desired_signal = current_size
                elif rsi_cross_up and fisher_recovering:
                    desired_signal = current_size * 0.8
                elif rsi_extreme_oversold and trend_12h_bullish:
                    desired_signal = current_size * 0.6
            
            # Short: Bearish trend + HMA cross or RSI pullback
            if trend_12h_bearish or hma_bearish:
                if hma_cross_down and rsi_overbought:
                    desired_signal = -current_size
                elif rsi_cross_down and fisher_weakening:
                    desired_signal = -current_size * 0.8
                elif rsi_extreme_overbought and trend_12h_bearish:
                    desired_signal = -current_size * 0.6
        
        # === RANGING REGIME LOGIC (CHOP > 55) ===
        elif ranging_regime:
            # Long: Fisher oversold + RSI oversold (mean reversion)
            if fisher_oversold and rsi_oversold:
                desired_signal = current_size
            elif fisher_cross_up and rsi_extreme_oversold:
                desired_signal = current_size * 0.7
            elif rsi_extreme_oversold:
                desired_signal = current_size * 0.5
            
            # Short: Fisher overbought + RSI overbought (mean reversion)
            if fisher_overbought and rsi_overbought:
                if desired_signal == 0:
                    desired_signal = -current_size
            elif fisher_cross_down and rsi_extreme_overbought:
                if desired_signal == 0:
                    desired_signal = -current_size * 0.7
            elif rsi_extreme_overbought and desired_signal == 0:
                desired_signal = -current_size * 0.5
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Conservative: require multiple confirmations
            if trend_12h_bullish and fisher_recovering and rsi_oversold:
                desired_signal = RANGE_SIZE * 0.7
            elif trend_12h_bearish and fisher_weakening and rsi_overbought:
                desired_signal = -RANGE_SIZE * 0.7
            elif hma_cross_up and rsi_cross_up:
                desired_signal = RANGE_SIZE * 0.5
            elif hma_cross_down and rsi_cross_down:
                desired_signal = -RANGE_SIZE * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and Fisher not overbought
                if (trend_12h_bullish or hma_bullish) and fisher_4h[i] < 1.5 and rsi_4h[i] < 70:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if trend intact and Fisher not oversold
                if (trend_12h_bearish or hma_bearish) and fisher_4h[i] > -1.5 and rsi_4h[i] > 30:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if strong bearish reversal
            if trend_12h_bearish and hma_bearish and fisher_4h[i] > 1.5:
                desired_signal = 0.0
            if ranging_regime and rsi_4h[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if strong bullish reversal
            if trend_12h_bullish and hma_bullish and fisher_4h[i] < -1.5:
                desired_signal = 0.0
            if ranging_regime and rsi_4h[i] < 25:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= TREND_SIZE:
                desired_signal = TREND_SIZE
            elif desired_signal >= RANGE_SIZE:
                desired_signal = RANGE_SIZE
            else:
                desired_signal = RANGE_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -TREND_SIZE:
                desired_signal = -TREND_SIZE
            elif desired_signal <= -RANGE_SIZE:
                desired_signal = -RANGE_SIZE
            else:
                desired_signal = -RANGE_SIZE * 0.5
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals