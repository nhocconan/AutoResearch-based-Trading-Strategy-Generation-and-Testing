# Strategy: 6h_VWAP_Deviation_12hEMA50_Trend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.326 | +10.6% | -6.0% | 50 | FAIL |
| ETHUSDT | 0.242 | +32.0% | -14.0% | 50 | PASS |
| SOLUSDT | -0.105 | +10.9% | -18.3% | 41 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.681 | +13.5% | -3.9% | 15 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
6h Volume-Weighted Average Price (VWAP) Deviation + 12h EMA50 Trend + Volume Spike
Hypothesis: Price deviations from VWAP tend to mean-revert in range markets but breakout in trending markets. 
Using 12h EMA50 to filter trend direction: long when price > VWAP + deviation AND above 12h EMA50, short when price < VWAP - deviation AND below 12h EMA50.
Volume spike confirms participation. Works in bull/bear by trend-filtering mean-reversion signals.
Target: 12-37 trades/year (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate VWAP (cumulative typical price * volume / cumulative volume)
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(volume)
    # Avoid division by zero
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # Calculate VWAP deviation (standard deviation of price-VWAP over 20 periods)
    price_vwap_diff = typical_price - vwap
    # Use pandas rolling for std with min_periods
    vwap_std = pd.Series(price_vwap_diff).rolling(window=20, min_periods=20).std().values
    
    # Deviation threshold: 2.0 * VWAP std
    deviation_upper = vwap + (2.0 * vwap_std)
    deviation_lower = vwap - (2.0 * vwap_std)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for VWAP std (20) + EMA50 warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(vwap[i]) or np.isnan(vwap_std[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_50_aligned[i]
        vwap_val = vwap[i]
        std_val = vwap_std[i]
        upper_band = deviation_upper[i]
        lower_band = deviation_lower[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Mean reversion signals with trend filter
        if position == 0:
            # Long: price below VWAP - deviation (oversold) AND above 12h EMA50 (uptrend filter)
            long_condition = (curr_close < lower_band) and (curr_close > ema_trend) and volume_spike
            # Short: price above VWAP + deviation (overbought) AND below 12h EMA50 (downtrend filter)
            short_condition = (curr_close > upper_band) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to VWAP or trend breaks
            if curr_close >= vwap_val or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to VWAP or trend breaks
            if curr_close <= vwap_val or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_VWAP_Deviation_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-25 01:06
