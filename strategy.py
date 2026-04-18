# NEW STRATEGY: 4h_Camilla_Pivot_R1S1_Breakout_Volume_Trend_Hybrid
# Hypothesis: Camarilla pivot levels R1/S1 from daily timeframe act as strong support/resistance.
# Breakouts above R1 or below S1 with volume confirmation and daily EMA trend filter capture
# institutional move initiation. Works in bull/bear by following institutional flow.
# Uses 4h primary timeframe with 1d HTF to target 20-50 trades/year (80-200 total over 4 years).
# Focus on BTC/ETH with volume confirmation to reduce false breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to 4h timeframe (waits for 1-day bar to close)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # 1-day EMA trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_4h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0
    bars_since_entry = 0
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or
            np.isnan(volume_filter[i]) or np.isnan(ema_1d_4h[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        price = close[i]
        r1_val = r1_4h[i]
        s1_val = s1_4h[i]
        vol_ok = volume_filter[i]
        ema_trend = ema_1d_4h[i]
        
        if position == 0:
            # Long: break above R1 with volume in uptrend
            if price > r1_val and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: break below S1 with volume in downtrend
            elif price < s1_val and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            bars_since_entry += 1
            # Minimum holding period: 4 bars (1 day)
            if bars_since_entry < 4:
                signals[i] = 0.25
            else:
                signals[i] = 0.25
                # Exit: price returns to S1 or trend reverses
                if price < s1_val or price < ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        
        elif position == -1:
            bars_since_entry += 1
            # Minimum holding period: 4 bars (1 day)
            if bars_since_entry < 4:
                signals[i] = -0.25
            else:
                signals[i] = -0.25
                # Exit: price returns to R1 or trend reverses
                if price > r1_val or price > ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "4h_Camilla_Pivot_R1S1_Breakout_Volume_Trend_Hybrid"
timeframe = "4h"
leverage = 1.0