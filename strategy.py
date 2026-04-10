#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend with 1d RSI filter and 1w volume spike confirmation
# - Primary: Kaufman Adaptive Moving Average (KAMA) direction on 12h timeframe
# - Entry filter: 1d RSI(14) between 30-70 (avoid extremes) + 1w volume > 2.0x 20-period volume MA
# - Exit: Price crosses KAMA in opposite direction
# - Position sizing: 0.25 (discrete level)
# - Works in bull/bear: KAMA adapts to market noise, volume filter ensures participation, RSI avoids exhaustion
# - Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe

name = "12h_1d_1w_kama_rsi_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate KAMA(10,2,30) on 12h close
    # ER = |net change| / sum(|abs change|)
    change = np.abs(np.diff(close, prepend=close[0]))
    direction = np.abs(np.diff(close, n=10, prepend=close[:10]))
    er = np.where(change > 0, direction / change, 0)
    # Smooth ER
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.copy(close)
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 1w volume confirmation: volume > 2.0x 20-period volume MA
    volume_ma_20_1w = pd.Series(volume_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    # Align HTF indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)  # KAMA calculated on 1d but we want 12h alignment
    # Actually recalculate KAMA on 12h data directly for simplicity
    # Recalculate KAMA on 12h close
    change_12h = np.abs(np.diff(close, prepend=close[0]))
    direction_12h = np.abs(np.diff(close, n=10, prepend=close[:10]))
    er_12h = np.where(change_12h > 0, direction_12h / change_12h, 0)
    sc_12h = (er_12h * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama_12h = np.copy(close)
    for i in range(1, n):
        kama_12h[i] = kama_12h[i-1] + sc_12h[i] * (close[i] - kama_12h[i-1])
    
    # Align 1d RSI to 12h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Align 1w volume MA to 12h
    vol_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    # Volume spike condition
    vol_spike = vol_1w_current > 2.0 * vol_ma_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_12h[i]) or np.isnan(rsi_1d_aligned[i]) or
            np.isnan(vol_1w_current[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: price vs KAMA + RSI filter + volume spike
        if position == 0:  # Flat - look for new entries
            # Long: price above KAMA + RSI not overbought + volume spike
            if close[i] > kama_12h[i] and rsi_1d_aligned[i] < 70 and vol_spike[i]:
                position = 1
                signals[i] = 0.25
            # Short: price below KAMA + RSI not oversold + volume spike
            elif close[i] < kama_12h[i] and rsi_1d_aligned[i] > 30 and vol_spike[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - exit when price crosses KAMA in opposite direction
            if position == 1:  # Long position
                if close[i] <= kama_12h[i]:  # Price crosses below KAMA
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= kama_12h[i]:  # Price crosses above KAMA
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals