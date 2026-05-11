# The strategy is designed for 12h timeframe with a focus on the 1d trend for higher timeframe bias and volume confirmation for entry confirmation.
# It uses Donchian breakouts aligned with the daily trend and volume spike to capture strong moves while minimizing false signals.
# The strategy aims for 12-37 trades per year by using tight entry conditions and volatility-based stops to manage risk.
# It avoids overtrading by requiring confluence of trend, breakout, and volume, which should reduce trade frequency and improve robustness across market regimes.

name = "12h_Donchian_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- 12h Donchian Channel (20-period) ---
    highest_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # --- Volume Filter: spike above 2.0x median of last 30 periods ---
    vol_median = pd.Series(volume_12h).rolling(window=30, min_periods=15).median().values
    vol_threshold = vol_median * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for Donchian and EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_12h[i] <= entry_price - 2.5 * (highest_high[i] - lowest_low[i]):
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] >= entry_price + 2.5 * (highest_high[i] - lowest_low[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1d trend
        trend_up = close_12h[i] > ema50_1d_aligned[i]
        trend_down = close_12h[i] < ema50_1d_aligned[i]
        
        # Volume filter: spike above 2.0x median
        vol_ok = volume_12h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume spike
            if close_12h[i] > highest_high[i] and trend_up and vol_ok:
                # Long: price breaks above Donchian high + 1d uptrend + volume spike
                signals[i] = 0.25
                position = 1
                entry_price = close_12h[i]
            elif close_12h[i] < lowest_low[i] and trend_down and vol_ok:
                # Short: price breaks below Donchian low + 1d downtrend + volume spike
                signals[i] = -0.25
                position = -1
                entry_price = close_12h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss: 2.5x ATR equivalent using Donchian width
                channel_width = highest_high[i] - lowest_low[i]
                if channel_width > 0:
                    if close_12h[i] <= entry_price - 2.5 * channel_width:
                        signals[i] = 0.0
                        position = 0
                    # Exit: price returns to or below midpoint of Donchian channel
                    elif close_12h[i] <= (highest_high[i] + lowest_low[i]) / 2:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss: 2.5x ATR equivalent using Donchian width
                channel_width = highest_high[i] - lowest_low[i]
                if channel_width > 0:
                    if close_12h[i] >= entry_price + 2.5 * channel_width:
                        signals[i] = 0.0
                        position = 0
                    # Exit: price returns to or above midpoint of Donchian channel
                    elif close_12h[i] >= (highest_high[i] + lowest_low[i]) / 2:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
                else:
                    signals[i] = -0.25
    
    return signals