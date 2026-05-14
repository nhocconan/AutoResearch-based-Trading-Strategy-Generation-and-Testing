# Strategy: 4h_Camarilla_H3L3_Breakout_1dEMA34_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.268 | +36.3% | -14.3% | 95 | PASS |
| ETHUSDT | 0.131 | +26.0% | -16.2% | 96 | PASS |
| SOLUSDT | 0.648 | +112.6% | -36.5% | 97 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.325 | +1.2% | -7.0% | 41 | FAIL |
| ETHUSDT | 0.800 | +23.5% | -8.8% | 30 | PASS |
| SOLUSDT | 0.489 | +16.6% | -11.7% | 28 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA34 trend filter (more stable for BTC/ETH trend identification).
- Entry: Long when price breaks above Camarilla H3 level AND price > 1d EMA34 AND volume > 2.0 * 4h volume MA(20);
         Short when price breaks below Camarilla L3 level AND price < 1d EMA34 AND volume > 2.0 * 4h volume MA(20).
- Exit: Close-based reversal (opposite signal) or stoploss via trend filter (signal=0 when price closes below/above 1d EMA34).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels provide structure for breakouts; 1d EMA34 filters counter-trend signals; volume confirmation avoids false breakouts.
- Works in bull markets (buy breakouts) and bear markets (sell breakdowns) with volume confirmation to avoid false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla H3 and L3 levels from previous day
    # Camarilla: H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    # We need previous day's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Get 1d data for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 4h data for volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    
    # Calculate volume MA(20) on 4h
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 1  # Need previous bar for Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Stoploss: exit if price closes below/above 1d EMA34 (trend filter)
        if position == 1:
            if curr_close < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            if curr_close > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions with volume confirmation and Camarilla breakout
        bullish_breakout = curr_high > camarilla_h3[i]  # Break above H3 level
        bearish_breakout = curr_low < camarilla_l3[i]   # Break below L3 level
        
        # Trend filter from 1d EMA34
        price_above_ema = curr_close > ema_34_aligned[i]
        price_below_ema = curr_close < ema_34_aligned[i]
        
        # Volume confirmation
        vol_confirm = curr_volume > 2.0 * vol_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Price breaks above Camarilla H3 AND price above 1d EMA34
                if bullish_breakout and price_above_ema:
                    signals[i] = 0.25
                    position = 1
                # Short: Price breaks below Camarilla L3 AND price below 1d EMA34
                elif bearish_breakout and price_below_ema:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-24 19:07
