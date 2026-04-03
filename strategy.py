#!/usr/bin/env python3
"""
Experiment #114: 1h RSI Mean Reversion + 4h/1d Trend Filter + Volume Spike

HYPOTHESIS: In 1h timeframe, RSI extremes (RSI<30 for long, RSI>70 for short) 
provide mean reversion opportunities when aligned with higher timeframe trend 
(4h EMA50 for direction, 1d EMA200 for regime filter) and confirmed by volume spikes.
Session filter (08-20 UTC) reduces noise. Discrete sizing (0.20) and ATR stop (2.0x) 
limit drawdown. Targets 15-37 trades/year on 1h to avoid fee drag. Works in bull/bear 
by fading extremes only when higher timeframe structure supports reversal.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_meanrev_4h_1d_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h EMA50 for trend direction ===
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === HTF: 1d EMA200 for regime filter (bull/bear market) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators ===
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0.0
    tr3[0] = 0.0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume MA(20) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20, mean).values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # Already datetime64[ns] index
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = -1
    
    warmup = 200  # For EMA200 and RSI stability
    
    for i in range(warmup, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(rsi_values[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if in_position:
                # Close position outside session
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = 0.0
            continue
        
        # --- Position Management ---
        if in_position:
            # ATR-based stoploss
            stop_distance = 2.0 * atr[i]
            if position_side > 0:  # Long
                if low[i] <= entry_price - stop_distance:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                else:
                    signals[i] = SIZE
            else:  # Short
                if high[i] >= entry_price + stop_distance:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry (Only if Flat) ---
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > vol_ma[i] * 1.5 if vol_ma[i] > 0 else False
        
        if not vol_ok:
            signals[i] = 0.0
            continue
        
        # Trend and regime filters
        hma_bullish = close[i] > ema_4h_aligned[i]  # Price above 4h EMA50 = short-term bullish
        regime_bullish = close[i] > ema_1d_aligned[i]  # Price above 1d EMA200 = bull regime
        
        # Long conditions: RSI oversold + bullish alignment
        if rsi_values[i] < 30 and hma_bullish and regime_bullish:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        
        # Short conditions: RSI overbought + bearish alignment
        elif rsi_values[i] > 70 and not hma_bullish and not regime_bullish:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        
        else:
            signals[i] = 0.0
    
    return signals