# Strategy: 6h_BollingerSqueeze_Breakout_RSI_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.000 | +0.0% | 0.0% | 0 | FAIL |
| ETHUSDT | 0.154 | +28.1% | -21.9% | 212 | PASS |
| SOLUSDT | 0.654 | +115.7% | -28.8% | 264 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.412 | +13.7% | -12.4% | 60 | PASS |
| SOLUSDT | 0.563 | +19.3% | -13.5% | 70 | PASS |

## Code
```python
#!/usr/bin/env python3
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
    
    # Get 1d data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Bollinger Bands (20, 2.0)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + (2.0 * std_20)
    bb_lower = sma_20 - (2.0 * std_20)
    
    # Bollinger Band Width for squeeze detection
    bb_width = (bb_upper - bb_lower) / sma_20
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_squeeze = bb_width < bb_width_ma * 0.8
    
    # 1d RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean().values
    avg_loss = loss.rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d Volume spike detection
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 2.0)
    
    # Align HTF indicators to 6h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    sma_20_aligned = align_htf_to_ltf(prices, df_1d, sma_20)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 80  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(sma_20_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(bb_squeeze_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Bollinger Band squeeze breakout conditions
        breakout_up = close[i] > bb_upper_aligned[i]
        breakout_down = close[i] < bb_lower_aligned[i]
        
        # Momentum filter: RSI in favorable range
        rsi_momentum_up = rsi_aligned[i] > 50
        rsi_momentum_down = rsi_aligned[i] < 50
        
        # Only trade during squeeze breakouts with momentum
        squeeze_active = bb_squeeze_aligned[i]
        vol_confirm = vol_spike_aligned[i]
        
        # Entry conditions - Bollinger Band squeeze breakout with volume
        long_entry = breakout_up and rsi_momentum_up and squeeze_active and vol_confirm
        short_entry = breakout_down and rsi_momentum_down and squeeze_active and vol_confirm
        
        # Exit conditions: return to middle Bollinger Band or opposite breakout
        long_exit = close[i] < sma_20_aligned[i] or breakout_down
        short_exit = close[i] > sma_20_aligned[i] or breakout_up
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_BollingerSqueeze_Breakout_RSI_Volume"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-28 06:13
