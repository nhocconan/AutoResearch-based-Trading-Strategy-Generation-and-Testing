# Strategy: 6h_EMA34_Trend_VolumeRSIFilter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.080 | +12.7% | -17.3% | 99 | FAIL |
| ETHUSDT | 0.122 | +25.3% | -15.7% | 105 | PASS |
| SOLUSDT | 1.111 | +255.9% | -26.7% | 106 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.474 | +15.2% | -9.7% | 28 | PASS |
| SOLUSDT | 0.721 | +23.0% | -8.7% | 20 | PASS |

## Code
```python
#!/usr/bin/env python3
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
    
    # Get daily data for trend, volatility, and price channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA(34) for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily ATR(14) for volatility
    tr1_d = df_1d['high'].values - df_1d['low'].values
    tr2_d = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3_d = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    tr_d[0] = tr1_d[0]
    atr_1d_raw = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_raw)
    
    # 6h ATR(14) for volatility filter
    tr1_h = high - low
    tr2_h = np.abs(high - np.roll(close, 1))
    tr3_h = np.abs(low - np.roll(close, 1))
    tr_h = np.maximum(tr1_h, np.maximum(tr2_h, tr3_h))
    tr_h[0] = tr1_h[0]
    atr_6h = pd.Series(tr_h).rolling(window=14, min_periods=14).mean().values
    
    # 6h RSI(14) for momentum confirmation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(atr_6h[i]) or 
            i >= len(atr_1d_aligned) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_1d_aligned[i]
        atr_6h_val = atr_6h[i]
        atr_1d_val = atr_1d_aligned[i]
        rsi_val = rsi[i]
        
        # Volatility filter: 6h ATR > 0.5 * daily ATR (higher volatility regime)
        vol_filter = atr_6h_val > (atr_1d_val * 0.5)
        
        # RSI filter: avoid overbought/oversold extremes
        rsi_filter = (rsi_val > 30) & (rsi_val < 70)
        
        if position == 0:
            # Long: price above EMA with volatility and RSI filter
            if close[i] > ema_trend and vol_filter and rsi_filter:
                signals[i] = size
                position = 1
            # Short: price below EMA with volatility and RSI filter
            elif close[i] < ema_trend and vol_filter and rsi_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA
            if close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above EMA
            if close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_EMA34_Trend_VolumeRSIFilter_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-27 06:18
