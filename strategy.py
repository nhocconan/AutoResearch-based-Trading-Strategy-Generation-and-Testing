#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wEMA34_Trend_VolumeSpike_ATRStop_v1
Hypothesis: Daily Camarilla R1/S1 breakouts with 1-week EMA34 trend filter and volume confirmation capture sustained momentum across market regimes. Uses ATR-based stoploss (2.0x) and minimum holding period (3 days) to reduce churn. Discrete sizing (0.25) targets 15-25 trades/year for BTC/ETH/SOL to minimize fee drag and improve generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend regime)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # === 1w EMA34 for trend regime ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Daily volume confirmation (volume > 1.8x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.8 * vol_ma_20)
    
    # === Daily Camarilla pivot levels (R1, S1) based on PREVIOUS day's OHLC ===
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # first day invalid
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12.0
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    days_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                days_since_entry = 0
            continue
        
        price = close[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        r1_val = r1[i]
        s1_val = s1[i]
        vol_conf = volume_confirmed[i]
        
        # Trend regime
        is_bull = price > ema_34_1w_val
        is_bear = price < ema_34_1w_val
        
        if position == 0:
            if is_bull:
                # Bull regime: long breakouts favored
                long_condition = (price > r1_val) and vol_conf
                short_condition = (price < s1_val) and vol_conf and (price < ema_34_1w_val * 0.99)  # stricter for shorts
            else:  # bear regime
                # Bear regime: short breakdowns favored
                short_condition = (price < s1_val) and vol_conf
                long_condition = (price > r1_val) and vol_conf and (price > ema_34_1w_val * 1.01)  # stricter for longs
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                days_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                days_since_entry = 0
        
        elif position != 0:
            days_since_entry += 1
            
            # Minimum holding period of 3 days to reduce churn
            if days_since_entry < 3:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    days_since_entry = 0
                # Exit if price breaks below S1 (failed breakout)
                elif price < s1_val:
                    signals[i] = 0.0
                    position = 0
                    days_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    days_since_entry = 0
                # Exit if price breaks above R1 (failed breakdown)
                elif price > r1_val:
                    signals[i] = 0.0
                    position = 0
                    days_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wEMA34_Trend_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0