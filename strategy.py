# 1h_4h_1d_volume_momentum_v1
# Hypothesis: Combine 4h trend direction with 1d momentum and volume surge on 1h for precise entries.
# Long when 4h close > 4h SMA(50) AND 1d RSI(14) > 50 AND 1h volume > 2x 20-period average AND 1h close > 1h SMA(20).
# Short when 4h close < 4h SMA(50) AND 1d RSI(14) < 50 AND 1h volume > 2x 20-period average AND 1h close < 1h SMA(20).
# Uses 4h for trend filter, 1d for momentum filter, 1h for entry timing with volume confirmation.
# Designed for 15-30 trades/year (~60-120 total over 4 years) to avoid fee drag.
# Works in bull markets via trend continuation and bear markets via counter-trend bounces in ranging conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_volume_momentum_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h SMA(20) for entry filter
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # 1h volume MA(20) for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    sma50_4h = pd.Series(df_4h['close']).rolling(window=50, min_periods=50).mean().values
    sma50_4h_aligned = align_htf_to_ltf(prices, df_4h, sma50_4h)
    
    # Get 1d data for momentum filter (RSI)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(sma20[i]) or np.isnan(vol_ma_20[i]) or np.isnan(sma50_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: trend breaks or momentum fades
            if close[i] < sma20[i] or df_4h['close'].iloc[i//4] < sma50_4h[i//4] if i >= 4 else False or rsi_1d_aligned[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: trend breaks or momentum fades
            if close[i] > sma20[i] or df_4h['close'].iloc[i//4] > sma50_4h[i//4] if i >= 4 else False or rsi_1d_aligned[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Get current 4h close for trend filter (using aligned data)
            # Note: Using aligned data ensures we only use closed 4h bars
            if i >= 4:  # Need at least 4 1h bars to get one 4h bar
                idx_4h = i // 4
                if idx_4h < len(df_4h):
                    trend_up = df_4h['close'].iloc[idx_4h] > sma50_4h[idx_4h]
                    trend_down = df_4h['close'].iloc[idx_4h] < sma50_4h[idx_4h]
                else:
                    trend_up = trend_down = False
            else:
                trend_up = trend_down = False
            
            # Long entry: uptrend + bullish momentum + volume surge + price above SMA20
            if trend_up and rsi_1d_aligned[i] > 50 and vol_surge and close[i] > sma20[i]:
                position = 1
                signals[i] = 0.20
            # Short entry: downtrend + bearish momentum + volume surge + price below SMA20
            elif trend_down and rsi_1d_aligned[i] < 50 and vol_surge and close[i] < sma20[i]:
                position = -1
                signals[i] = -0.20
    
    return signals