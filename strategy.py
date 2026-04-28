#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA20 pullback strategy with 4h EMA50 trend filter and 1d volume spike confirmation
# Uses 4h EMA50 for intermediate trend direction. Enters on 1h EMA20 pullbacks in trend direction
# with 1d volume spike confirmation for institutional interest. Target: 15-30 trades/year via tight
# pullback conditions + volume + trend filter. Works in bull (trend continuation) and bear (trend retracements).

name = "1h_EMA20_Pullback_4hEMA50_Trend_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close for trend filter
    close_4h = pd.Series(df_4h['close'])
    ema50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume MA20 for spike confirmation
    volume_1d = pd.Series(df_1d['volume'])
    volume_ma20_1d = volume_1d.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1h EMA20 for pullback entries
    close_series = pd.Series(close)
    ema20_1h = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 4h EMA50 to 1h timeframe (completed 4h candles only)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Align 1d volume MA20 to 1h timeframe (completed 1d candles only)
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    # Volume spike: current 1h volume > 2.0 * aligned 1d volume MA20
    volume_spike = volume > 2.0 * volume_ma20_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 50)  # EMA20 and EMA50 need sufficient history
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_ma20_1d_aligned[i]) or 
            np.isnan(ema20_1h[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        ema20_val = ema20_1h[i]
        ema50_val = ema50_4h_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price pulls back to EMA20 in uptrend AND volume spike
            if price >= ema20_val * 0.998 and price <= ema20_val * 1.002 and price > ema50_val and vol_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short entry: price pulls back to EMA20 in downtrend AND volume spike
            elif price >= ema20_val * 0.998 and price <= ema20_val * 1.002 and price < ema50_val and vol_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or trend reversal
            # ATR-based stoploss: 2.5 * ATR below entry (using 1h ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.5 * atr_val
            # Exit on stoploss or price < EMA50 (trend breakdown)
            if price < stop_loss or price < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit on stoploss or trend reversal
            # ATR-based stoploss: 2.5 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.5 * atr_val
            # Exit on stoploss or price > EMA50 (trend breakdown)
            if price > stop_loss or price > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals