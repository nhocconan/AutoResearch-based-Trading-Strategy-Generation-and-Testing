# Strategy: 12h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.458 | +7.8% | -7.4% | 141 | FAIL |
| ETHUSDT | 0.282 | +32.0% | -7.3% | 126 | PASS |
| SOLUSDT | 0.368 | +44.6% | -25.2% | 127 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.405 | +11.1% | -4.7% | 52 | PASS |
| SOLUSDT | -0.701 | -2.5% | -11.6% | 45 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla H3/L3 levels with volume confirmation and 1d EMA(34) trend filter
# Camarilla H3/L3 represent strong intraday support/resistance where institutional order flow clusters.
# Breakouts above H3 or below L3 with volume spike indicate strong institutional participation.
# 1d EMA(34) ensures alignment with longer-term trend to avoid counter-trend trades.
# Designed for low trade frequency (<30/year) to minimize fee drag in both bull and bear markets.
# Uses 12h timeframe as requested, with 1d HTF for Camarilla levels and trend filter.

name = "12h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (H3, L3, H4, L4)
    # Based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point (PP)
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Calculate Camarilla levels
    h3 = pp + (high_1d - low_1d) * 1.1 / 4.0
    l3 = pp - (high_1d - low_1d) * 1.1 / 4.0
    h4 = pp + (high_1d - low_1d) * 1.1 / 2.0
    l4 = pp - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d_s = pd.Series(close_1d)
    ema_34_1d = close_1d_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # warmup for EMA(34)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (2.0 * vol_ma_20)
        
        curr_close = close[i]
        curr_ema = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        curr_h3 = h3_aligned[i]
        curr_l3 = l3_aligned[i]
        curr_h4 = h4_aligned[i]
        curr_l4 = l4_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 1d Camarilla H3 with 1d uptrend
                if curr_close > curr_h3 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 1d Camarilla L3 with 1d downtrend
                elif curr_close < curr_l3 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 1d Camarilla L3
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_l3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1d Camarilla H4
            elif curr_close >= curr_h4:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 1d Camarilla H3
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_h3:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1d Camarilla L4
            elif curr_close <= curr_l4:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-30 10:29
