# Strategy: 4h_Donchian_Breakout_Volume_12hTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.037 | +21.8% | -12.6% | 152 | KEEP |
| ETHUSDT | 0.645 | +58.8% | -12.9% | 134 | KEEP |
| SOLUSDT | 0.453 | +58.2% | -28.6% | 127 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.319 | -4.9% | -5.8% | 59 | DISCARD |
| ETHUSDT | 0.754 | +17.1% | -5.8% | 49 | KEEP |
| SOLUSDT | -0.049 | +4.7% | -11.7% | 44 | DISCARD |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian channel breakout with volume confirmation and 12h trend filter.
# Donchian(20) provides clear breakout levels. Volume > 2x 20-period average confirms breakout strength.
# 12h EMA(50) trend filter ensures alignment with higher timeframe trend to avoid counter-trend entries.
# Long when price breaks above upper Donchian with volume and 12h bullish.
# Short when price breaks below lower Donchian with volume and 12h bearish.
# Exit when price returns to middle of Donchian channel or trend changes.
# Designed for ~20-30 trades/year with strict entry conditions to avoid overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channel (20-period high/low)
    high_max = np.full(len(high_4h), np.nan)
    low_min = np.full(len(low_4h), np.nan)
    
    for i in range(19, len(high_4h)):
        high_max[i] = np.max(high_4h[i-19:i+1])
        low_min[i] = np.min(low_4h[i-19:i+1])
    
    # Middle of channel
    mid_channel = (high_max + low_min) / 2.0
    
    # Align Donchian levels to 4h timeframe (already at 4h, but need to align to lower timeframe if needed)
    # Since we're using 4h as primary, we'll use the 4h data directly but need to handle alignment properly
    # For simplicity in this case, we'll work with 4h index and then align to lower timeframe if needed
    # But since we're using 4h as primary timeframe, we need to get lower timeframe data
    
    # Actually, let's use 4h as the primary timeframe for signal generation
    # We need to get the lower timeframe data (e.g., 1h) for entry timing
    # But the instructions say to use timeframe = "4h", so we'll generate signals at 4h frequency
    
    # Let me reconsider - the primary timeframe is 4h, so we should generate signals at 4h intervals
    # But we need to work with the prices array which is at the execution timeframe
    
    # Let's restart with clearer approach
    
    # Get 1h data for entry timing (lower timeframe than 4h)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    volume_1h = df_1h['volume'].values
    
    # Calculate Donchian channel on 1h data (20-period)
    high_max_1h = np.full(len(high_1h), np.nan)
    low_min_1h = np.full(len(low_1h), np.nan)
    
    for i in range(19, len(high_1h)):
        high_max_1h[i] = np.max(high_1h[i-19:i+1])
        low_min_1h[i] = np.min(low_1h[i-19:i+1])
    
    mid_channel_1h = (high_max_1h + low_min_1h) / 2.0
    
    # Align Donchian levels to the execution timeframe (prices array)
    # Since prices is at execution timeframe (let's assume it's 1h or lower), we need to align
    # But we don't know the execution timeframe from the prices array alone
    
    # Let's assume the prices array is at 1h timeframe for now
    # In practice, the backtesting engine will pass data at the strategy's timeframe
    
    # Actually, let's simplify and work at 4h level throughout
    # We'll get 4h data and generate signals at 4h frequency
    
    # Re-implementing with clearer logic:
    
    # Get execution timeframe data (this is what the backtesting engine provides)
    # We'll assume it's at least 1h resolution
    
    # For 4h strategy, we want to use 4h for signals but can use lower timeframe for entry timing
    # However, to keep it simple and avoid alignment issues, let's use 4h throughout
    
    # Let's use 4h as the working timeframe
    
    # Get 4h OHLCV data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian channel (20-period) on 4h data
    high_max_4h = np.full(len(high_4h), np.nan)
    low_min_4h = np.full(len(low_4h), np.nan)
    
    for i in range(19, len(high_4h)):
        high_max_4h[i] = np.max(high_4h[i-19:i+1])
        low_min_4h[i] = np.min(low_4h[i-19:i+1])
    
    mid_channel_4h = (high_max_4h + low_min_4h) / 2.0
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h Donchian levels and 12h EMA to execution timeframe
    # We need to align to the prices array timeframe
    
    # Since we don't know the execution timeframe, let's assume we're working at 4h level
    # and the prices array is at 4h frequency
    
    # For now, let's assume the prices array is at 4h frequency
    # In practice, we would need to handle the alignment properly
    
    # Let's take a different approach - use the prices array directly for execution
    # and use higher timeframes for filtering
    
    # Reset and implement cleanly:
    
    # Use prices array as our execution timeframe (could be 1h, 30m, etc.)
    # Use 4h for Donchian channel calculation
    # Use 12h for trend filter
    
    # We already have df_4h and df_12h from above
    
    # Calculate Donchian on 4h data
    high_max_4h = np.full(len(high_4h), np.nan)
    low_min_4h = np.full(len(low_4h), np.nan)
    
    for i in range(19, len(high_4h)):
        high_max_4h[i] = np.max(high_4h[i-19:i+1])
        low_min_4h[i] = np.min(low_4h[i-19:i+1])
    
    mid_channel_4h = (high_max_4h + low_min_4h) / 2.0
    
    # Calculate 12h EMA
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h channels and 12h EMA to execution timeframe
    high_max_aligned = align_htf_to_ltf(prices, df_4h, high_max_4h)
    low_min_aligned = align_htf_to_ltf(prices, df_4h, low_min_4h)
    mid_channel_aligned = align_htf_to_ltf(prices, df_4h, mid_channel_4h)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: volume > 2x 20-period average (execution timeframe)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 4h Donchian (20), 12h EMA (50), volume MA (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_max_aligned[i]) or np.isnan(low_min_aligned[i]) or 
            np.isnan(mid_channel_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Trend filters from 12h EMA
        bullish_trend = price > ema_50_aligned[i]
        bearish_trend = price < ema_50_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume and bullish trend
            if price > high_max_aligned[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian with volume and bearish trend
            elif price < low_min_aligned[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns below middle of channel or trend turns bearish
            if price < mid_channel_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns above middle of channel or trend turns bullish
            if price > mid_channel_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_Volume_12hTrend"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-27 10:43
