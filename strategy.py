#!/usr/bin/env python3
"""
Experiment #023: 6h Volume Imbalance Reversal + ATR Regime

HYPOTHESIS: Volume imbalance (taker_buy_volume/total_volume) combined with 
volatility expansion captures institutional moves without requiring exact 
price levels like Donchian.

KEY DIFFERENCE from failed strategies:
- Donchian requires price to reach specific level → 0 trades
- Volume imbalance triggers on volume profile changes → regular occurrences
- ATR ratio confirms volatility expansion → filters noise

WHY IT WORKS IN BOTH BULL AND BEAR:
- Bull: Taker buy imbalance + ATR expansion = institutional accumulation
- Bear: Same setup = weak longs trapped, reversal to short
- 1d EMA filter ensures we fade against the correct trend direction

EXPECTED TRADES: 50-130 over 4 years (12-33/year) — within target range.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_vol_imbalance_atr_regime_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy = prices["taker_buy_volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 for trend direction
    sma50_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # 1d ATR for relative filtering
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d Volume MA for comparison
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === 6h Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    # ATR ratio: current volatility vs recent
    atr_ratio = atr_14 / np.where(atr_30 > 0, atr_30, 1)
    
    # Volume imbalance: taker buy / total volume
    vol_imbalance = taker_buy / np.where(volume > 0, volume, 1)
    
    # Volume MA (20 bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI(14) for momentum
    def calc_rsi(prices_arr, period=14):
        n = len(prices_arr)
        deltas = np.zeros(n)
        deltas[1:] = prices_arr[1:] - prices_arr[:-1]
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
        avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
        
        rs = avg_gain / np.where(avg_loss > 0, avg_loss, 1)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_14 = calc_rsi(close, period=14)
    
    # EMA(21) for local trend
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # === Signal Generation ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    # Minimum holding period to avoid fee churn (6h bars, so 3 bars = 18h)
    MIN_HOLD = 3
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Conditions:
            # 1. Volume imbalance > 0.58 (aggressive buying)
            # 2. ATR ratio > 1.3 (volatility expanding)
            # 3. RSI not overbought (< 65)
            # 4. 1d trend: price above SMA50 (bull bias)
            
            long_vol_imbalance = vol_imbalance[i] > 0.58
            long_atr_expansion = atr_ratio[i] > 1.3
            long_rsi_ok = rsi_14[i] < 65
            long_1d_bull = (close[i] > sma50_1d_aligned[i]) if not np.isnan(sma50_1d_aligned[i]) else True
            
            if long_vol_imbalance and long_atr_expansion and long_rsi_ok and long_1d_bull:
                desired_signal = SIZE
                
            # === SHORT ENTRY ===
            # Conditions:
            # 1. Volume imbalance < 0.42 (aggressive selling)
            # 2. ATR ratio > 1.3 (volatility expanding)
            # 3. RSI not oversold (> 35)
            # 4. 1d trend: price below SMA50 (bear bias)
            
            short_vol_imbalance = vol_imbalance[i] < 0.42
            short_atr_expansion = atr_ratio[i] > 1.3
            short_rsi_ok = rsi_14[i] > 35
            short_1d_bear = (close[i] < sma50_1d_aligned[i]) if not np.isnan(sma50_1d_aligned[i]) else False
            
            if short_vol_imbalance and short_atr_expansion and short_rsi_ok and short_1d_bear:
                desired_signal = -SIZE
        
        # === STOPLOSS AND TRAILING EXIT ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: 3 ATR from highest point
                stop_price = trailing_high - 3.0 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if 1d trend turns bearish
                if close[i] < sma50_1d_aligned[i] if not np.isnan(sma50_1d_aligned[i]) else False:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Take profit at 2.5R
                profit_target = entry_price + 2.5 * entry_atr
                if high[i] >= profit_target:
                    # Trail stop at 1.5 ATR from peak
                    trail_stop = trailing_high - 1.5 * atr_14[i]
                    if close[i] < trail_stop:
                        desired_signal = SIZE / 2  # Reduce to half
                        in_position = False
                        position_side = 0
                        
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: 3 ATR from lowest point
                stop_price = trailing_low + 3.0 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if 1d trend turns bullish
                if close[i] > sma50_1d_aligned[i] if not np.isnan(sma50_1d_aligned[i]) else True:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Take profit at 2.5R
                profit_target = entry_price - 2.5 * entry_atr
                if low[i] <= profit_target:
                    # Trail stop at 1.5 ATR from bottom
                    trail_stop = trailing_low + 1.5 * atr_14[i]
                    if close[i] > trail_stop:
                        desired_signal = -SIZE / 2  # Reduce to half
                        in_position = False
                        position_side = 0
        
        # === MINIMUM HOLD PERIOD ===
        if in_position and (i - entry_bar) < MIN_HOLD:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        
        signals[i] = desired_signal
    
    return signals