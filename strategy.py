#!/usr/bin/env python3
"""
1d_KAMA_Trend_Regime_Filter_v1
Hypothesis: On daily timeframe, Kaufman Adaptive Moving Average (KAMA) identifies the primary trend regime. 
Price above KAMA indicates bull regime (favor longs on pullbacks to value), price below KAMA indicates bear regime (favor shorts on rallies to value). 
Combined with RSI(2) for mean-reversion entries within the trend and volume confirmation to ensure participation. 
Discrete sizing (0.25) minimizes fee churn. Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for weekly trend context)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1d KAMA (10,2,30) for daily trend regime ===
    close = prices['close'].values
    direction = np.abs(np.diff(close, 10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, 1)), axis=0)  # 10-period sum of absolute changes
    # Handle first 10 values
    direction = np.concatenate([np.full(10, np.nan), direction])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, direction / volatility, 0)  # Efficiency Ratio
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # Smoothing Constant
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Seed
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_aligned = kama  # Already 1d aligned
    
    # === 1w EMA34 for weekly trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily RSI(2) for mean-reversion entries ===
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Wilder's smoothing
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[1] = gain[1]
    avg_loss[1] = loss[1]
    for i in range(2, n):
        avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2
        avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Daily volume confirmation (volume > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 20  # max 20 days
    
    for i in range(50, n):  # Warmup for indicators
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        rsi_val = rsi[i]
        vol_conf = volume_confirmed[i]
        
        # Weekly trend regime filter
        weekly_bull = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-5]  # Rising weekly EMA
        weekly_bear = ema_34_1w_aligned[i] < ema_34_1w_aligned[i-5]  # Falling weekly EMA
        
        if position == 0:
            # Bull regime: price above KAMA + rising weekly EMA
            if price > kama_val and weekly_bull:
                # Long on RSI pullback (<30) with volume
                if rsi_val < 30 and vol_conf:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    bars_since_entry = 0
            # Bear regime: price below KAMA + falling weekly EMA
            elif price < kama_val and weekly_bear:
                # Short on RSI rally (>70) with volume
                if rsi_val > 70 and vol_conf:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Exit conditions
            if position == 1:  # Long
                # Exit on RSI mean-reversion (>50) or weekly trend change
                if rsi_val > 50 or not weekly_bull:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short
                # Exit on RSI mean-reversion (<50) or weekly trend change
                if rsi_val < 50 or not weekly_bear:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_Regime_Filter_v1"
timeframe = "1d"
leverage = 1.0