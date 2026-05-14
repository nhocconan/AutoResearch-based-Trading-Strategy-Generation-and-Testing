# Strategy: 12h_ParabolicSAR_Volume_ADX

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.258 | +5.4% | -12.5% | 143 | FAIL |
| ETHUSDT | 0.372 | +45.4% | -14.6% | 132 | PASS |
| SOLUSDT | 0.714 | +110.0% | -34.1% | 127 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.006 | +4.9% | -12.1% | 44 | PASS |
| SOLUSDT | -0.498 | -5.5% | -23.5% | 39 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Wilder's Parabolic SAR with 1-day volume confirmation and ADX trend filter.
# Long when: PSAR flips below price, ADX(1d) > 25, volume > 1.5x 20-period average
# Short when: PSAR flips above price, ADX(1d) > 25, volume > 1.5x 20-period average
# Exit when PSAR flips to opposite side.
# Parabolic SAR is designed to capture trends with built-in acceleration, working in both bull and bear markets.
# Target: 15-25 trades/year per symbol. Uses Wilder's original smoothing for consistency.
name = "12h_ParabolicSAR_Volume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on daily data using Wilder's smoothing
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing function
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period]) 
        # Subsequent values
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_period = 14
    atr = wilders_smoothing(tr, atr_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, atr_period)
    
    # Directional Indicators
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, atr_period)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Parabolic SAR on 12h data (Wilder's original)
    # Initial values
    psar = np.full(n, np.nan)
    trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
    af = np.full(n, 0.02)  # acceleration factor
    ep = np.full(n, 0.0)   # extreme point
    
    # Initialize with first 2 bars
    if n >= 2:
        if close[1] > close[0]:
            trend[0] = 1
            psar[0] = low[0]
            ep[0] = high[1]
        else:
            trend[0] = -1
            psar[0] = high[0]
            ep[0] = low[1]
    
    # Calculate PSAR for each bar
    for i in range(1, n):
        if i == 1:
            psar[i] = psar[0] + af[0] * (ep[0] - psar[0])
        else:
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
        
        # Reverse trend if price crosses PSAR
        if trend[i-1] == 1 and low[i] <= psar[i]:
            trend[i] = -1
            psar[i] = ep[i-1]  # SAR becomes prior EP
            ep[i] = low[i]
            af[i] = 0.02
        elif trend[i-1] == -1 and high[i] >= psar[i]:
            trend[i] = 1
            psar[i] = ep[i-1]  # SAR becomes prior EP
            ep[i] = high[i]
            af[i] = 0.02
        else:
            trend[i] = trend[i-1]
            # Update EP and AF
            if trend[i] == 1:
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + 0.02, 0.2)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
            else:
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + 0.02, 0.2)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        
        # Ensure SAR stays within prior period's range
        if trend[i] == 1:
            psar[i] = min(psar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
        else:
            psar[i] = max(psar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_aligned[i]) or np.isnan(psar[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: price above PSAR (uptrend), ADX > 25, volume confirmation
            if price > psar[i] and adx_val > 25 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: price below PSAR (downtrend), ADX > 25, volume confirmation
            elif price < psar[i] and adx_val > 25 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below PSAR (trend reversal)
            if price <= psar[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above PSAR (trend reversal)
            if price >= psar[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 00:47
