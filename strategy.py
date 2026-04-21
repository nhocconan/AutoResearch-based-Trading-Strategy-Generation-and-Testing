#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dRegime_v1
Hypothesis: Use 4h EMA50 for trend direction and 1d volatility regime (ATR ratio) to filter Camarilla breakouts on 1h.
In high volatility regime (expanding markets), trade breakouts with trend alignment.
In low volatility regime (contracting markets), avoid breakouts to reduce false signals.
Uses discrete sizing (0.20) and session filter (08-20 UTC) to target 15-30 trades/year.
Works in bull/bear via volatility regime adaptation: breakouts in high vol, stricter in low vol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend, 1d for volatility regime)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 20:
        return np.zeros(n)
    
    # === 4h EMA50 for trend alignment ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1d ATR ratio for volatility regime (ATR(10) / ATR(30)) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1_1d = pd.Series(high_1d - low_1d)
    tr2_1d = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3_1d = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    
    atr10_1d = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean().values
    atr30_1d = pd.Series(tr_1d).rolling(window=30, min_periods=30).mean().values
    atr_ratio_1d = atr10_1d / atr30_1d
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # High volatility regime: ATR ratio > 1.0 (expanding volatility)
    high_vol_regime = atr_ratio_aligned > 1.0
    
    # === 1h data for entry timing ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1h ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 1h volume confirmation (volume > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    # === 1h Camarilla pivot levels (R1, S1) based on PREVIOUS bar's OHLC ===
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # first bar invalid
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12.0
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12.0
    
    # Session filter: 08-20 UTC (avoid Asian session noise)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(volume_confirmed[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_50_4h_val = ema_50_4h_aligned[i]
        r1_val = r1[i]
        s1_val = s1[i]
        vol_conf = volume_confirmed[i]
        high_vol = high_vol_regime[i]
        
        # Trend alignment: price above/below 4h EMA50
        uptrend = price > ema_50_4h_val
        downtrend = price < ema_50_4h_val
        
        if position == 0:
            # Entry conditions: Camarilla breakout with trend and volume confirmation
            # In high volatility regime: standard breakout
            # In low volatility regime: require stronger breakout (price > R1/S1 by 0.2%)
            if high_vol:
                # High volatility: breakout strategy
                long_condition = (price > r1_val) and uptrend and vol_conf
                short_condition = (price < s1_val) and downtrend and vol_conf
            else:
                # Low volatility: require stronger breakout to avoid false signals
                long_condition = (price > r1_val * 1.002) and uptrend and vol_conf
                short_condition = (price < s1_val * 0.998) and downtrend and vol_conf
            
            if long_condition:
                signals[i] = 0.20
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.20
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 4 bars to reduce churn
            if bars_since_entry < 4:
                signals[i] = 0.20 if position == 1 else -0.20
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks below S1 (failed breakout) or trend deteriorates
                elif price < s1_val or price < ema_50_4h_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if price breaks above R1 (failed breakdown) or trend deteriorates
                elif price > r1_val or price > ema_50_4h_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dRegime_v1"
timeframe = "1h"
leverage = 1.0