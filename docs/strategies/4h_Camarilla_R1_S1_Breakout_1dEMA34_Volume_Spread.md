# Strategy: 4h_Camarilla_R1_S1_Breakout_1dEMA34_Volume_Spread

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.463 | +44.0% | -10.9% | 267 | PASS |
| ETHUSDT | 0.026 | +20.0% | -15.1% | 257 | PASS |
| SOLUSDT | 0.763 | +104.5% | -25.0% | 251 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.181 | +7.7% | -5.5% | 90 | PASS |
| ETHUSDT | 0.490 | +13.4% | -12.6% | 95 | PASS |
| SOLUSDT | 0.173 | +8.1% | -11.9% | 86 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Volume_Spread"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla R1 and S1 levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    r1 = close_prev + 1.1 * (high_prev - low_prev) / 12
    s1 = close_prev - 1.1 * (high_prev - low_prev) / 12
    
    # Align daily levels to 4h timeframe (with 1-day delay for completed bar)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d trend filter: EMA34
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    # Additional spread filter: avoid trading when spread is too wide (proxy: high-low > 2*ATR)
    # Simple proxy: avoid extreme volatility days
    atr_period = 14
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    volatility_filter = (high - low) < (2 * atr)  # Avoid excessively volatile bars
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~8 hours for 4h to reduce trades
    
    start_idx = max(100, 20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 1d trend direction
        trend_up = close > ema_34_1d_aligned[i]
        trend_down = close < ema_34_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Break above R1 in uptrend with volume and reasonable volatility
            if (close[i] > r1_aligned[i] and 
                trend_up[i] and 
                vol_filter[i] and
                volatility_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Break below S1 in downtrend with volume and reasonable volatility
            elif (close[i] < s1_aligned[i] and 
                  trend_down[i] and 
                  vol_filter[i] and
                  volatility_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price re-enters Camarilla body (between R1 and S1) or trend change
            if (close[i] < r1_aligned[i] and close[i] > s1_aligned[i]) or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price re-enters Camarilla body or trend change
            if (close[i] < r1_aligned[i] and close[i] > s1_aligned[i]) or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 breakouts with 1d EMA34 trend alignment capture institutional
# breakouts in both bull and bear markets. Volume filter ensures genuine participation,
# volatility filter avoids choppy conditions. Spread between R1/S1 provides natural
# target zone. Conservative sizing (0.25) limits drawdown. Target: 20-35 trades/year.
```

## Last Updated
2026-05-07 12:07
