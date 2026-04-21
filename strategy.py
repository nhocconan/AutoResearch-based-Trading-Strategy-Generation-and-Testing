#!/usr/bin/env python3
"""
1h_4hTrend_1dVolumeRegime_Entry_v1
Hypothesis: Use 4h EMA50 for trend direction, 1d volume regime (high/low volume) to filter noise, and 1h RSI(2) for precise entry timing. In 4h uptrend + high volume regime, look for RSI(2) < 10 pullbacks to go long. In 4h downtrend + low volume regime, look for RSI(2) > 90 bounces to go short. Designed for 1h timeframe with tight entry conditions to avoid overtrading. Works in bull (trend + volume) and bear (counter-trend in low volume) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend, 1d for volume regime)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h EMA50 for trend direction ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1d volume regime: high volume if volume > 1.5x 20-day MA, low volume if < 0.5x ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 1h RSI(2) for entry timing ===
    close = prices['close'].values
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_50_4h_val = ema_50_4h_aligned[i]
        vol_ma_20_1d_val = vol_ma_20_1d_aligned[i]
        volume_1d_current = df_1d['volume'].iloc[min(i // 24, len(df_1d)-1)] if i // 24 < len(df_1d) else volume_1d[-1]
        rsi_val = rsi[i]
        
        # Determine regimes
        is_4h_uptrend = price > ema_50_4h_val
        is_4h_downtrend = price < ema_50_4h_val
        is_high_volume = volume_1d_current > (1.5 * vol_ma_20_1d_val)
        is_low_volume = volume_1d_current < (0.5 * vol_ma_20_1d_val)
        
        if position == 0:
            # Long conditions: 4h uptrend + high volume + RSI(2) < 10 (oversold pullback)
            long_condition = is_4h_uptrend and is_high_volume and (rsi_val < 10)
            
            # Short conditions: 4h downtrend + low volume + RSI(2) > 90 (overbought bounce)
            short_condition = is_4h_downtrend and is_low_volume and (rsi_val > 90)
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions
            if position == 1:  # long
                # Exit if RSI(2) > 80 (overbought) or trend breaks
                if rsi_val > 80 or price < ema_50_4h_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1, short
                # Exit if RSI(2) < 20 (oversold) or trend breaks
                if rsi_val < 20 or price > ema_50_4h_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_4hTrend_1dVolumeRegime_Entry_v1"
timeframe = "1h"
leverage = 1.0