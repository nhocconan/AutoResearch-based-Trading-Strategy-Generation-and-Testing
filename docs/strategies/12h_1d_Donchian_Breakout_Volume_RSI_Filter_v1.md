# Strategy: 12h_1d_Donchian_Breakout_Volume_RSI_Filter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.002 | -5.4% | -14.3% | 41 | FAIL |
| ETHUSDT | 0.277 | +31.4% | -6.0% | 39 | PASS |
| SOLUSDT | 0.018 | +18.8% | -18.9% | 42 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.426 | +11.4% | -7.2% | 16 | PASS |
| SOLUSDT | -0.795 | -4.8% | -14.2% | 12 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with daily volume confirmation and RSI filter.
# Uses daily Donchian channels and RSI to avoid whipsaws in both bull and bear markets.
# Timeframe 12h reduces trade frequency to minimize fee drag while capturing medium-term trends.
# Volume confirmation ensures breakouts have conviction. RSI filter avoids overextended moves.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within profitable range.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 20-day Donchian channel on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily RSI for overbought/oversold conditions
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate daily volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 12-hour timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(rsi_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 12h volume > 1.5x daily volume MA (adjusted for 12h)
        # 2 12h periods per day, so daily MA/2 = approximate 12h period MA
        volume_12h_approx_ma = volume_ma_20_1d_aligned[i] / 2
        volume_condition = volume[i] > (volume_12h_approx_ma * 1.5)
        
        # RSI conditions: avoid extreme overbought/oversold
        rsi_not_overbought = rsi_aligned[i] < 70
        rsi_not_oversold = rsi_aligned[i] > 30
        
        # Entry conditions: Donchian breakout with volume and RSI filter
        # Long when price breaks above Donchian high with volume and not overbought
        # Short when price breaks below Donchian low with volume and not oversold
        breakout_long = close[i] > donchian_high_aligned[i]
        breakout_short = close[i] < donchian_low_aligned[i]
        
        if position == 0:
            if breakout_long and volume_condition and rsi_not_overbought:
                position = 1
                signals[i] = position_size
            elif breakout_short and volume_condition and rsi_not_oversold:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price breaks below Donchian low or RSI becomes overbought
            if close[i] < donchian_low_aligned[i] or rsi_aligned[i] >= 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price breaks above Donchian high or RSI becomes oversold
            if close[i] > donchian_high_aligned[i] or rsi_aligned[i] <= 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_Breakout_Volume_RSI_Filter_v1"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-13 20:44
