#!/usr/bin/env python3
"""
Experiment #756: 12h Primary + 1d HTF — Simplified Regime + HMA Trend + RSI Pullback

Hypothesis: After analyzing 500+ failed strategies and the success of #751 (Sharpe=0.342 at 4h):
1. 12h timeframe needs SIMPLER logic than 4h — fewer conflicting conditions
2. 1d HMA(21) provides reliable trend bias (proven in #749, #751)
3. RSI(14) pullback to 35-40 in uptrend, 60-65 in downtrend captures entries
4. Choppiness Index(14) filters trend vs range but with wider thresholds for 12h
5. Looser entry thresholds ensure >=30 trades/train (common failure mode)
6. ATR(14) trailing stop 2.5x protects against adverse moves
7. Discrete signals: 0.0, ±0.25, ±0.30 to minimize fee churn

Strategy design:
1. 1d HMA(21) for primary trend bias (aligned via mtf_data helper)
2. 12h RSI(14) for entry timing (pullback entries)
3. 12h Choppiness Index(14) for regime detection
4. 12h ATR(14) for trailing stop
5. Simple 2-regime logic: trending (CHOP<50) vs ranging (CHOP>50)

Key differences from #751:
- 12h primary instead of 4h (fewer trades, higher quality)
- Simpler RSI thresholds (35/65 instead of CRSI extremes)
- Single Choppiness threshold (50 instead of 38.2/61.8)
- Wider entry windows to ensure trade frequency on 12h

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 12h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_chop_simple_1d_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures whether market is trending or ranging.
    CHOP > 50 = ranging (mean reversion)
    CHOP < 50 = trending (trend follow)
    Simplified threshold for 12h timeframe.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (12h) indicators
    rsi_12h = calculate_rsi(close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    sma_50 = calculate_sma(close, period=50)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
        if np.isnan(rsi_12h[i]) or np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(chop_12h[i]):
            continue
        
        # === TREND BIAS (1d HTF HMA) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        trending_regime = chop_12h[i] < 50
        ranging_regime = chop_12h[i] >= 50
        
        # === RSI SIGNALS (12h RSI) ===
        rsi_oversold = rsi_12h[i] < 40
        rsi_overbought = rsi_12h[i] > 60
        rsi_extreme_low = rsi_12h[i] < 30
        rsi_extreme_high = rsi_12h[i] > 70
        rsi_neutral = 35 <= rsi_12h[i] <= 65
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        above_sma200 = close[i] > sma_200[i]
        below_sma50 = close[i] < sma_50[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        
        # === TRENDING REGIME LOGIC (CHOP < 50) ===
        if trending_regime:
            # Long: 1d bullish + RSI pullback + above SMA50
            if trend_1d_bullish and rsi_oversold and above_sma50:
                desired_signal = BASE_SIZE
            
            # Short: 1d bearish + RSI rally + below SMA50
            if trend_1d_bearish and rsi_overbought and below_sma50:
                desired_signal = -BASE_SIZE
            
            # Strong trend continuation (RSI neutral but trend strong)
            if trend_1d_bullish and above_sma50 and above_sma200 and rsi_neutral:
                desired_signal = BASE_SIZE
            
            if trend_1d_bearish and below_sma50 and below_sma200 and rsi_neutral:
                desired_signal = -BASE_SIZE
        
        # === RANGING REGIME LOGIC (CHOP >= 50) ===
        elif ranging_regime:
            # Mean reversion long: RSI extreme low + 1d not bearish
            if rsi_extreme_low and not trend_1d_bearish:
                desired_signal = REDUCED_SIZE
            
            # Mean reversion short: RSI extreme high + 1d not bullish
            if rsi_extreme_high and not trend_1d_bullish:
                desired_signal = -REDUCED_SIZE
            
            # Range bounce with trend bias
            if rsi_oversold and trend_1d_bullish:
                desired_signal = REDUCED_SIZE
            
            if rsi_overbought and trend_1d_bearish:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1d still bullish and RSI not overbought
                if trend_1d_bullish and rsi_12h[i] < 70:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if 1d still bearish and RSI not oversold
                if trend_1d_bearish and rsi_12h[i] > 30:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses or RSI extremely overbought
            if trend_1d_bearish and rsi_12h[i] > 65:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses or RSI extremely oversold
            if trend_1d_bullish and rsi_12h[i] < 35:
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
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
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