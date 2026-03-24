#!/usr/bin/env python3
"""
Experiment #811: 4h Primary + 1d/1w HTF — Relaxed Regime Switch + RSI + Donchian

Hypothesis: After 553 failed strategies, key insights:
1. 4h timeframe balances trade frequency (20-50/year) with fee drag
2. Previous strategies failed due to TOO STRICT entry conditions (0 trades)
3. Simplifying CRSI to standard RSI(14) reduces overfitting
4. Wider Choppiness thresholds (50/40 vs 55/45) generate more regime switches
5. Shorter Donchian period (15 vs 20) captures more breakouts
6. RELAXED RSI thresholds (35/65 vs 20/80) ensure >=10 trades per symbol
7. 1d HMA(21) + 1w HMA(21) dual trend filter for robust direction
8. ATR trailing stop at 2.0x (tighter than 2.5x) for faster exits

Strategy design:
1. 1d HMA(21) for intermediate trend (aligned via mtf_data)
2. 1w HMA(21) for long-term trend bias (aligned via mtf_data)
3. 4h Choppiness Index(14) for regime detection
4. 4h RSI(14) for entry timing (simpler than CRSI, more robust)
5. 4h Donchian(15) for breakout entries in trending regime
6. 4h ATR(14) for trailing stop (2.0x)
7. Discrete signals: 0.0, ±0.20, ±0.30
8. RELAXED thresholds to guarantee trades on ALL symbols

Key changes from #807 (1d) that failed:
- 4h primary instead of 1d (more trades, same fee efficiency with HTF filter)
- RSI(14) instead of CRSI (simpler, less overfitting)
- RSI thresholds: 35/65 (not 20/80) — generates 3x more trades
- CHOP thresholds: 50/40 (not 55/45) — more regime switches
- Donchian period: 15 (not 20) — more breakout signals
- Dual HTF trend (1d + 1w HMA) — more robust than single 1w

Target: Sharpe > 0.612, trades >= 15 train, >= 5 test, ALL symbols positive
Timeframe: 4h (target 25-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_relaxed_regime_rsi_donchian_1d1w_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 50 = ranging, CHOP < 40 = trending.
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

def calculate_donchian(high, low, period=15):
    """Donchian Channels — highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

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
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=15)
    atr_4h = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate and align 1d HMA for intermediate trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for long-term trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(chop_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === LONG-TERM TREND BIAS (1w HTF HMA21) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === SECULAR TREND FILTER (SMA200) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === REGIME DETECTION (4h Choppiness Index) ===
        ranging_regime = chop_4h[i] > 50
        trending_regime = chop_4h[i] < 40
        neutral_regime = not ranging_regime and not trending_regime
        
        # === RSI SIGNALS (RELAXED for more trades) ===
        rsi_oversold = rsi_4h[i] < 35
        rsi_overbought = rsi_4h[i] > 65
        rsi_extreme_oversold = rsi_4h[i] < 25
        rsi_extreme_overbought = rsi_4h[i] > 75
        rsi_neutral_low = 35 <= rsi_4h[i] < 50
        rsi_neutral_high = 50 < rsi_4h[i] <= 65
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        desired_signal = 0.0
        
        # === RANGING REGIME LOGIC (CHOP > 50) — Mean Reversion ===
        if ranging_regime:
            # Long: RSI oversold + trend alignment (relaxed: any 1 filter)
            if rsi_oversold and (above_sma200 or trend_1d_bullish or trend_1w_bullish):
                desired_signal = BASE_SIZE
            
            # Short: RSI overbought + trend alignment
            if rsi_overbought and (below_sma200 or trend_1d_bearish or trend_1w_bearish):
                desired_signal = -BASE_SIZE
            
            # Conservative: extreme RSI alone (guarantees trades)
            if rsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME LOGIC (CHOP < 40) — Trend Following ===
        elif trending_regime:
            # Long: 1w bullish + Donchian breakout (relaxed: 1d also ok)
            if (trend_1w_bullish or trend_1d_bullish) and donchian_breakout_long:
                desired_signal = BASE_SIZE
            
            # Short: 1w bearish + Donchian breakout
            if (trend_1w_bearish or trend_1d_bearish) and donchian_breakout_short:
                desired_signal = -BASE_SIZE
            
            # Pullback entries in trend (relaxed RSI)
            if (trend_1w_bullish or trend_1d_bullish) and rsi_neutral_low and above_sma200:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if (trend_1w_bearish or trend_1d_bearish) and rsi_neutral_high and below_sma200:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === NEUTRAL REGIME (40 <= CHOP <= 50) ===
        else:
            # Conservative: RSI extremes + any trend alignment
            if rsi_extreme_oversold and (trend_1w_bullish or trend_1d_bullish or above_sma200):
                desired_signal = REDUCED_SIZE
            
            if rsi_extreme_overbought and (trend_1w_bearish or trend_1d_bearish or below_sma200):
                desired_signal = -REDUCED_SIZE
            
            # Basic mean reversion with single filter
            if rsi_oversold and above_sma200:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            if rsi_overbought and below_sma200:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if any trend intact and RSI not overbought
                if (trend_1w_bullish or trend_1d_bullish or above_sma200) and rsi_4h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if any trend intact and RSI not oversold
                if (trend_1w_bearish or trend_1d_bearish or below_sma200) and rsi_4h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if all trends reverse + RSI overbought
            if trend_1w_bearish and trend_1d_bearish and rsi_4h[i] > 70:
                desired_signal = 0.0
            # Exit if RSI extremely overbought in ranging regime
            if ranging_regime and rsi_4h[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if all trends reverse + RSI oversold
            if trend_1w_bullish and trend_1d_bullish and rsi_4h[i] < 30:
                desired_signal = 0.0
            # Exit if RSI extremely oversold in ranging regime
            if ranging_regime and rsi_4h[i] < 25:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
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