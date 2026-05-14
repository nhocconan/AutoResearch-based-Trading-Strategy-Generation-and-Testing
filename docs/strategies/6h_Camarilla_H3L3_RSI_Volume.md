# Strategy: 6h_Camarilla_H3L3_RSI_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.146 | +27.1% | -10.0% | 142 | PASS |
| ETHUSDT | 0.370 | +43.6% | -18.1% | 124 | PASS |
| SOLUSDT | 0.634 | +92.6% | -26.8% | 107 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.820 | -3.3% | -8.7% | 51 | FAIL |
| ETHUSDT | 0.857 | +21.0% | -8.2% | 42 | PASS |
| SOLUSDT | -0.523 | -4.3% | -19.9% | 38 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Camarilla pivot levels (H3/L3) from 1-day timeframe combined with volume spike and RSI(14) momentum.
# Camarilla H3/L3 act as intraday support/resistance; breakouts with volume indicate strong momentum.
# RSI(14) filters for momentum alignment to avoid false breakouts in chop.
# Designed for 6h timeframe to capture medium-term breakouts with low frequency.
# Entry: Long when close > H3 and RSI > 50 and volume spike; Short when close < L3 and RSI < 50 and volume spike.
# Exit: Opposite Camarilla level touch (H3 for long exit, L3 for short exit) or RSI reversal.
# Uses strict conditions to limit trades (~15-25/year) and avoid overtrading.
name = "6h_Camarilla_H3L3_RSI_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Camarilla pivot levels (based on prior day OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: H3/L3 = close ± (high-low)*1.1/2
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align to 6h timeframe (waits for prior day close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above H3 with bullish momentum and volume
            if (close[i] > camarilla_h3_aligned[i] and 
                rsi[i] > 50 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below L3 with bearish momentum and volume
            elif (close[i] < camarilla_l3_aligned[i] and 
                  rsi[i] < 50 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches L3 or RSI turns bearish
            if (close[i] < camarilla_l3_aligned[i]) or (rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches H3 or RSI turns bullish
            if (close[i] > camarilla_h3_aligned[i]) or (rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 19:48
