# 4h_1d_RSI_Trend_12hADX_TrendFilter_v1
# 4h RSI mean-reversion with 12h ADX trend filter and volume confirmation
# Long when RSI(14) < 30 and price near 4h Bollinger lower band + volume spike + 12h ADX > 25
# Short when RSI(14) > 70 and price near 4h Bollinger upper band + volume spike + 12h ADX > 25
# Exit when RSI returns to 50 or ADX < 20
# Works in both bull and bear by using ADX trend filter (only trade strong trends)
# Target: 25-50 trades/year (100-200 total over 4 years) to minimize fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_RSI_Trend_12hADX_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Get 1d data for volume calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate RSI(14) on 4h data
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate Bollinger Bands (20, 2) on 4h data
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    bb_upper_values = bb_upper.values
    bb_lower_values = bb_lower.values
    
    # Calculate ADX(14) on 12h data
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = df_12h['high'].diff()
    minus_dm = df_12h['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_values)
    
    # Volume spike filter: 1d volume > 2x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_values[i]) or np.isnan(bb_upper_values[i]) or 
            np.isnan(bb_lower_values[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 2x 1d average volume (scaled)
        # Scale 1d average to 4h: 1d has 6x 4h bars, so divide by 6
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 2.0 * (vol_ma_1d_aligned[i] / 6.0)
        
        if position == 0:
            # Look for long entry: RSI oversold + price near BB lower + volume spike + strong trend
            if (rsi_values[i] < 30 and 
                close[i] <= bb_lower_values[i] * 1.02 and  # Near or below lower band
                volume_filter and 
                adx_12h_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Look for short entry: RSI overbought + price near BB upper + volume spike + strong trend
            elif (rsi_values[i] > 70 and 
                  close[i] >= bb_upper_values[i] * 0.98 and  # Near or above upper band
                  volume_filter and 
                  adx_12h_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when RSI returns to 50 or trend weakens
            if (rsi_values[i] >= 50 or 
                adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when RSI returns to 50 or trend weakens
            if (rsi_values[i] <= 50 or 
                adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals