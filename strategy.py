#!/usr/bin/env python3
"""
EXPERIMENT #018 - MTF 1h HMA Trend + RSI Pullback + Volume Confirmation + ATR Stop
====================================================================================
Hypothesis: Combining 4h HMA trend filter with 1h RSI pullback entries, volume 
confirmation, and proper ATR-based trailing stops will reduce whipsaws while 
capturing major trends. Volume spikes confirm genuine breakouts vs fake moves.

Key improvements:
- 4h HMA(21) for trend direction (smoother than EMA, less lag than SMA)
- 1h RSI(14) pullback to 40-60 zone for entries (not extremes)
- Volume spike filter (1.5x 20-bar avg) confirms momentum
- ATR(14) trailing stop sets signal=0 when price moves 2.5*ATR against position
- Discrete position sizing (0.0, ±0.30) minimizes fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_4h_hma_rsi_volume_atr_v1"
timeframe = "1h"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    hull = 2 * wma1 - wma2
    hma = hull.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_atr(high, low, close, period=14):
    """Calculate ATR with proper min_periods"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes above threshold * rolling avg"""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    spike = (volume > threshold * vol_avg.values).astype(float)
    return spike


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP (Rule 1) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA trend
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF with proper shift (Rule 2 - no manual i//N)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # === CALCULATE 1h INDICATORS (vectorized before loop) ===
    hma_1h = calculate_hma(close, 21)
    rsi_1h = calculate_rsi(close, 14)
    atr_1h = calculate_atr(high, low, close, 14)
    vol_spike = calculate_volume_spike(volume, 20, 1.5)
    
    # Calculate price relative to 4h HMA for trend strength
    hma_4h_pct = (close - hma_4h_aligned) / hma_4h_aligned * 100
    
    # === GENERATE SIGNALS ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (discrete, max 0.40)
    
    # Track position state for stoploss
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period for all indicators
    warmup = max(50, int(np.sqrt(21)) + 21 + 14)
    
    for i in range(warmup, n):
        # Skip if any indicator is NaN
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]):
            signals[i] = 0.0
            continue
        
        # === TREND FILTER (4h HMA) ===
        # Only long if price > 4h HMA, only short if price < 4h HMA
        trend_long = close[i] > hma_4h_aligned[i] and hma_4h_aligned[i] > hma_4h_aligned[i-1]
        trend_short = close[i] < hma_4h_aligned[i] and hma_4h_aligned[i] < hma_4h_aligned[i-1]
        
        # === ENTRY CONDITIONS (1h RSI pullback + volume) ===
        # Long: RSI pulled back to 40-55 zone in uptrend + volume spike
        long_entry = (trend_long and 
                      40 <= rsi_1h[i] <= 55 and 
                      vol_spike[i] == 1.0)
        
        # Short: RSI rallied to 45-60 zone in downtrend + volume spike
        short_entry = (trend_short and 
                       45 <= rsi_1h[i] <= 60 and 
                       vol_spike[i] == 1.0)
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # ATR-based trailing stop: exit if price moves 2.5*ATR against position
        stoploss_distance = 2.5 * atr_1h[i]
        
        if position_side == 1:  # Long position
            # Update highest price since entry
            if entry_price == 0.0 or close[i] > highest_since_entry:
                highest_since_entry = close[i]
            
            # Trailing stop: exit if price drops from highest
            trailing_stop = highest_since_entry - stoploss_distance
            
            # Hard stoploss from entry
            hard_stop = entry_price - stoploss_distance
            
            if close[i] < max(hard_stop, trailing_stop):
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
        
        elif position_side == -1:  # Short position
            # Update lowest price since entry
            if entry_price == 0.0 or close[i] < lowest_since_entry:
                lowest_since_entry = close[i]
            
            # Trailing stop: exit if price rises from lowest
            trailing_stop = lowest_since_entry + stoploss_distance
            
            # Hard stoploss from entry
            hard_stop = entry_price + stoploss_distance
            
            if close[i] > min(hard_stop, trailing_stop):
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === GENERATE SIGNAL ===
        if long_entry and position_side != 1:
            signals[i] = SIZE
            position_side = 1
            entry_price = close[i]
            highest_since_entry = close[i]
        elif short_entry and position_side != -1:
            signals[i] = -SIZE
            position_side = -1
            entry_price = close[i]
            lowest_since_entry = close[i]
        elif position_side == 1 and not trend_long:
            # Trend reversed, exit long
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
            highest_since_entry = 0.0
        elif position_side == -1 and not trend_short:
            # Trend reversed, exit short
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
            lowest_since_entry = 0.0
        else:
            # Hold current position
            signals[i] = signals[i-1] if i > 0 else 0.0
    
    return signals