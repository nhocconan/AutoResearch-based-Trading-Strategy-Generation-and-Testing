# Strategy: 4h_1d_camarilla_breakout_volume_trend_atrstop_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.774 | +47.7% | -4.0% | 317 | PASS |
| ETHUSDT | 0.291 | +32.5% | -6.9% | 311 | PASS |
| SOLUSDT | 0.144 | +27.2% | -21.0% | 259 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.842 | +0.5% | -4.3% | 125 | FAIL |
| ETHUSDT | 0.880 | +16.3% | -8.2% | 126 | PASS |
| SOLUSDT | 1.233 | +20.6% | -4.8% | 85 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla breakout with volume confirmation and 1d trend filter + ATR stoploss
# - Long when price breaks above Camarilla H3 level with volume > 1.8x 20-bar average AND 1d close > 1d EMA50
# - Short when price breaks below Camarilla L3 level with volume > 1.8x 20-bar average AND 1d close < 1d EMA50
# - Exit when price retreats to Camarilla H4/L4 levels OR ATR-based stoploss hit
# - Uses 1d trend filter to avoid counter-trend trades and ATR stoploss for risk control
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 20-40 trades/year on 4h timeframe (80-160 total over 4 years)
# - Focus on BTC/ETH; SOL-only strategies are low value

name = "4h_1d_camarilla_breakout_volume_trend_atrstop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute volume confirmation: > 1.8x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_20_avg)
    
    # Pre-compute volume filter for exit: < 0.8x average volume (loss of momentum)
    vol_weak = prices['volume'] < (0.8 * volume_20_avg)
    
    # Pre-compute ATR(14) for stoploss
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    tr = np.maximum(np.maximum(high_low, high_close), low_close)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0  # track entry price for stoploss
    
    # Pre-compute aligned 1d data
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    h_1d_aligned = align_htf_to_ltf(prices, df_1d, h_1d)
    l_1d_aligned = align_htf_to_ltf(prices, df_1d, l_1d)
    c_1d_aligned = align_htf_to_ltf(prices, df_1d, c_1d)
    
    # Pre-compute 1d EMA(50) for trend filter
    ema50_1d = pd.Series(c_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(atr[i]) or np.isnan(h_1d_aligned[i]) or np.isnan(l_1d_aligned[i]) or 
            np.isnan(c_1d_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get previous completed 1d bar values for Camarilla calculation
        # Since 4h timeframe, 1d data updates every 6 bars (24h/4h = 6)
        # Look back to the previous multiple of 6 to get completed 1d bar
        lookback_idx = (i // 6) * 6  # Start of current 1d bar
        if lookback_idx >= 6:  # Need at least one previous completed 1d bar
            prev_1d_idx = lookback_idx - 6  # Previous completed 1d bar
            
            if prev_1d_idx >= 0:
                ph = h_1d_aligned[prev_1d_idx]  # Previous 1d high
                pl = l_1d_aligned[prev_1d_idx]  # Previous 1d low
                pc = c_1d_aligned[prev_1d_idx]  # Previous 1d close
                
                # Calculate Camarilla levels
                range_val = ph - pl
                if range_val > 0:
                    camarilla_h3 = pc + (range_val * 1.1 / 4)
                    camarilla_l3 = pc - (range_val * 1.1 / 4)
                    camarilla_h4 = pc + (range_val * 1.1 / 2)
                    camarilla_l4 = pc - (range_val * 1.1 / 2)
                    
                    if position == 0:  # Flat - look for new breakout entries
                        # Long breakout: price > Camarilla H3 with volume spike AND 1d uptrend
                        if (prices['close'].iloc[i] > camarilla_h3 and 
                            vol_spike.iloc[i] and 
                            prices['close'].iloc[i] > ema50_1d_aligned[i]):
                            position = 1
                            entry_price = prices['close'].iloc[i]
                            signals[i] = 0.25
                        # Short breakdown: price < Camarilla L3 with volume spike AND 1d downtrend
                        elif (prices['close'].iloc[i] < camarilla_l3 and 
                              vol_spike.iloc[i] and 
                              prices['close'].iloc[i] < ema50_1d_aligned[i]):
                            position = -1
                            entry_price = prices['close'].iloc[i]
                            signals[i] = -0.25
                    else:  # Have position - look for exit
                        # Exit conditions:
                        # 1. Price retreats to Camarilla H4/L4 levels
                        # 2. Volume drops below 0.8x average (loss of momentum)
                        # 3. ATR-based stoploss hit
                        exit_signal = False
                        if position == 1:  # Long position
                            if (prices['close'].iloc[i] < camarilla_h4 or 
                                vol_weak.iloc[i] or
                                prices['close'].iloc[i] < entry_price - 2.5 * atr[i]):
                                exit_signal = True
                        elif position == -1:  # Short position
                            if (prices['close'].iloc[i] > camarilla_l4 or 
                                vol_weak.iloc[i] or
                                prices['close'].iloc[i] > entry_price + 2.5 * atr[i]):
                                exit_signal = True
                        
                        if exit_signal:
                            position = 0
                            entry_price = 0.0
                            signals[i] = 0.0
                        else:
                            if position == 1:
                                signals[i] = 0.25
                            else:
                                signals[i] = -0.25
                else:
                    # Hold current position
                    if position == 0:
                        signals[i] = 0.0
                    elif position == 1:
                        signals[i] = 0.25
                    else:
                        signals[i] = -0.25
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Not enough data yet, hold flat
            signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-04-10 05:06
