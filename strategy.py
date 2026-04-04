#!/usr/bin/env python3
"""
Experiment #5280: 4h Donchian(20) Breakout + HMA(21) Trend + Volume Confirmation + ATR Stoploss
HYPOTHESIS: On 4h timeframe, price breaking above/below the 20-period Donchian channel with HMA(21) trend alignment and volume > 1.5x average provides high-probability entries. In bull regime (price > 1d EMA50), we go long on upper breakout; in bear regime (price < 1d EMA50), we go short on lower breakout. Uses discrete position sizing (0.25) to balance profit potential with drawdown control. Designed for 19-50 trades/year on 4h timeframe (75-200 total over 4 years) to minimize fee drag. Works in bull markets by catching strong uptrends and in bear markets by catching strong downtrends, while avoiding ranging conditions where Donchian channel is narrow.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5280_4h_donchian20_hma_vol_1d_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for regime filter (EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        ema_50 = pd.Series(df_1d['close']).ewm(span=50, min_periods=50, adjust=False).mean().shift(1).values
        ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    else:
        ema_50_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: HMA(21) ===
    def hma(arr, period):
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean().values
        wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean().values
        raw = 2 * wma2 - wma1
        return pd.Series(raw).ewm(span=sqrt, adjust=False).mean().values
    
    hma_21 = hma(close, 21)
    
    # === 4h Indicators: Volume Average (20) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    stop_price = 0.0
    
    warmup = max(20, 21, 20, 50, 14)  # Donchian, HMA, Vol MA, EMA50, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(hma_21[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Stoploss or reversal ---
        if in_position:
            # Check stoploss
            if position_side > 0:  # Long position
                if price <= stop_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Check for HMA trend reversal (price crosses below HMA)
                if price < hma_21[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                if price >= stop_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Check for HMA trend reversal (price crosses above HMA)
                if price > hma_21[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Continue holding position
            signals[i] = SIZE if position_side > 0 else -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout
        breakout_up = price > highest_high[i-1]  # Use previous bar's channel
        breakout_down = price < lowest_low[i-1]   # Use previous bar's channel
        
        # HMA trend filter
        hma_bullish = price > hma_21[i]
        hma_bearish = price < hma_21[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Regime filter from 1d
        regime_bullish = price > ema_50_aligned[i]
        regime_bearish = price < ema_50_aligned[i]
        
        # Entry conditions
        if breakout_up and hma_bullish and vol_confirm and regime_bullish:
            in_position = True
            position_side = 1
            entry_price = close[i]
            stop_price = entry_price - 2.0 * atr[i]  # 2*ATR stoploss
            signals[i] = SIZE
        elif breakout_down and hma_bearish and vol_confirm and regime_bearish:
            in_position = True
            position_side = -1
            entry_price = close[i]
            stop_price = entry_price + 2.0 * atr[i]  # 2*ATR stoploss
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals