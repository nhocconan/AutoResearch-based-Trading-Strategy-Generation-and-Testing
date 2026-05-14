# Strategy: 4h_Camarilla_H3L3_12hEMA50_Trend_VolumeSpike_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.156 | +15.7% | -6.9% | 283 | FAIL |
| ETHUSDT | 0.039 | +22.0% | -11.9% | 265 | PASS |
| SOLUSDT | -0.061 | +14.3% | -17.4% | 213 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.319 | +22.3% | -7.2% | 99 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA50 trend filter and volume spike.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h EMA50 for trend filter (price above/below EMA50 defines bull/bear regime).
- Entry: Long when price breaks above Camarilla H3 in bull regime with volume > 2.0 * 4h volume MA(20);
         Short when price breaks below Camarilla L3 in bear regime with volume > 2.0 * 4h volume MA(20).
- Exit: ATR trailing stop (2.5 * ATR(14)) or opposite Camarilla breakout.
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Camarilla levels provide institutional pivot points, EMA50 filter avoids counter-trend trades,
  volume spike ensures strong participation. Works in bull (breakouts with trend) and bear (strong moves after panic lows/highs).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h volume MA(20) for confirmation
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Calculate 4h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels (H3, L3) on 4h data using previous day's OHLC
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    # Using 4h bar's OHLC to calculate levels for next bar
    camarilla_h3 = close + 1.1 * (high - low) / 2
    camarilla_l3 = close - 1.1 * (high - low) / 2
    # Shift to avoid look-ahead: levels calculated from current bar apply to next bar
    camarilla_h3 = np.roll(camarilla_h3, 1)
    camarilla_l3 = np.roll(camarilla_l3, 1)
    camarilla_h3[0] = np.nan
    camarilla_l3[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0
    lowest_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14, 1)  # EMA50 needs 50, volume MA needs 20, ATR needs 14, plus 1 for roll
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_4h_aligned[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: 2.0x threshold (tight to reduce trades)
        vol_spike = curr_volume > 2.0 * vol_ma_4h_aligned[i]
        
        # Trend filter: price above/below 12h EMA50
        bull_regime = curr_close > ema_50_12h_aligned[i]
        bear_regime = curr_close < ema_50_12h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above Camarilla H3 in bull regime with volume spike
            if curr_close > camarilla_h3[i] and bull_regime and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: price breaks below Camarilla L3 in bear regime with volume spike
            elif curr_close < camarilla_l3[i] and bear_regime and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position: update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: ATR trailing stop or opposite breakout (below L3)
            if curr_low <= highest_since_entry - 2.5 * atr[i] or curr_close < camarilla_l3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: ATR trailing stop or opposite breakout (above H3)
            if curr_high >= lowest_since_entry + 2.5 * atr[i] or curr_close > camarilla_h3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-24 16:47
