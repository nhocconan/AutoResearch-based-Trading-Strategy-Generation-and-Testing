#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrendRegime_VolumeSpike_v1
Hypothesis: On 1h timeframe, Camarilla R3/S3 breakouts combined with 4h EMA34 trend regime and volume confirmation (volume > 1.8x 20-period average) captures high-probability directional moves. 
In bull regime (4h close > 4h EMA34), favor longs on breaks above R3; in bear regime (4h close < 4h EMA34), favor shorts on breaks below S3. 
Volume confirmation ensures institutional participation. Discrete sizing (0.20) minimizes fee churn. Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend regime)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h EMA34 for trend regime ===
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === 1h Camarilla levels (based on previous day) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Camarilla levels using previous day's OHLC
    # We'll use rolling window of 24 bars (1 day on 1h) for previous day's OHLC
    prev_day_high = pd.Series(high).rolling(window=24, min_periods=24).max().shift(24).values
    prev_day_low = pd.Series(low).rolling(window=24, min_periods=24).min().shift(24).values
    prev_day_close = pd.Series(close).rolling(window=24, min_periods=24).last().shift(24).values
    
    # Camarilla levels
    range_ = prev_day_high - prev_day_low
    camarilla_r3 = prev_day_close + (range_ * 1.1 / 4)
    camarilla_s3 = prev_day_close - (range_ * 1.1 / 4)
    
    # === 1h volume confirmation (volume > 1.8x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.8 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 24  # max 1 day (24 * 1h = 24h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        regime_4h = ema_34_4h_aligned[i]
        camarilla_r3_val = camarilla_r3[i]
        camarilla_s3_val = camarilla_s3[i]
        vol_conf = volume_confirmed[i]
        
        # Trend regime
        is_bull = price > regime_4h
        is_bear = price < regime_4h
        
        if position == 0:
            if is_bull:
                # Bull regime: long when price breaks above R3
                long_condition = (price > camarilla_r3_val) and vol_conf
            else:  # bear regime
                # Bear regime: short when price breaks below S3
                short_condition = (price < camarilla_s3_val) and vol_conf
            
            if is_bull and long_condition:
                signals[i] = 0.20
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif is_bear and short_condition:
                signals[i] = -0.20
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Check stoploss (2.5x ATR)
            atr_val = np.abs(high[i] - low[i])  # Simple ATR proxy
            if position == 1:
                if price < entry_price - 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if price > entry_price + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit
                elif bars_since_entry >= max_hold_bars:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hTrendRegime_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0