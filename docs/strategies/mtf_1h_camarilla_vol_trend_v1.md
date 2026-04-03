# Strategy: mtf_1h_camarilla_vol_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.429 | +166.4% | -8.8% | 344 | PASS |
| SOLUSDT | 2.360 | +881.9% | -22.1% | 353 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.929 | +47.8% | -8.2% | 111 | PASS |
| SOLUSDT | 1.531 | +40.1% | -9.3% | 116 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #034: 1h Camarilla Pivot + Volume Spike + 4h/1d Trend Filter

HYPOTHESIS: Camarilla pivot levels on 1h identify intraday support/resistance. 
Combined with 4h trend filter (price > EMA50 for longs, < EMA50 for shorts) and 
1d volume confirmation (>1.8x average), this strategy captures breakouts with 
institutional participation. The 1h timeframe provides entry timing precision 
while 4h/1d filters reduce false signals. Uses discrete position sizing (0.20) 
and session filter (08-20 UTC) to minimize fee drag. Target: 60-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_camarilla_vol_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate EMA(50) on 4h close
    if len(df_4h) >= 50:
        close_4h = df_4h['close'].values
        ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    else:
        ema_50_4h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume spike confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 1h Camarilla Pivot Levels (using previous day's OHLC) ===
    # Calculate daily OHLC from 1h data
    daily_high = np.full(n, np.nan)
    daily_low = np.full(n, np.nan)
    daily_close = np.full(n, np.nan)
    
    if n >= 24:  # Need at least 24 hours for previous day
        for i in range(24, n):
            # Get previous day's 24-hour window (24 bars back)
            start_idx = i - 24
            end_idx = i
            daily_high[i] = np.max(high[start_idx:end_idx])
            daily_low[i] = np.min(low[start_idx:end_idx])
            daily_close[i] = close[end_idx - 1]  # Previous bar's close
    
    # Calculate Camarilla levels
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    if n >= 24:
        for i in range(24, n):
            if not (np.isnan(daily_high[i]) or np.isnan(daily_low[i]) or np.isnan(daily_close[i])):
                range_val = daily_high[i] - daily_low[i]
                if range_val > 0:
                    camarilla_h3[i] = daily_close[i] + range_val * 1.1 / 4
                    camarilla_l3[i] = daily_close[i] - range_val * 1.1 / 4
                    camarilla_h4[i] = daily_close[i] + range_val * 1.1 / 2
                    camarilla_l4[i] = daily_close[i] - range_val * 1.1 / 2
                else:
                    camarilla_h3[i] = daily_close[i]
                    camarilla_l3[i] = daily_close[i]
                    camarilla_h4[i] = daily_close[i]
                    camarilla_l4[i] = daily_close[i]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(100, 24)  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in alignment with 4h EMA50 ---
        price_above_4h_ema = close[i] > ema_50_4h_aligned[i]
        price_below_4h_ema = close[i] < ema_50_4h_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long when price breaks above H3 with volume and trend alignment
        long_condition = (
            close[i] > camarilla_h3[i] and 
            volume_spike and 
            price_above_4h_ema
        )
        
        # Short when price breaks below L3 with volume and trend alignment
        short_condition = (
            close[i] < camarilla_l3[i] and 
            volume_spike and 
            price_below_4h_ema
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
```

## Last Updated
2026-04-03 10:33
