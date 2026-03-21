#!/usr/bin/env python3
"""
EXPERIMENT #033 - KAMA Adaptive Trend + RSI Pullback + Dual HTF Filter (1h primary)
====================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts better to volatility 
regimes than fixed EMA/HMA. Combined with RSI pullback entries (45-55 zone) and 
dual HTF trend confirmation (4h + 12h), this should reduce whipsaws while 
capturing sustained trends. Volume spike filter confirms breakout validity.

Key features:
- Primary TF: 1h (required for this experiment)
- HTF filters: 4h KAMA(21) + 12h KAMA(50) for multi-scale trend alignment
- Trend: KAMA(21) on 1h with efficiency ratio adaptation
- Entry: RSI(14) pullback to 45-55 zone in trend direction
- Volume confirmation: volume > 1.5x 20-period average
- Regime: ADX(14) > 25 for trending market filter
- Stoploss: 2.5*ATR(14) trailing
- Position sizing: 0.25 base, discrete levels (0.0, ±0.125, ±0.25)

Why this differs from failed attempts:
- KAMA adapts to volatility (unlike fixed HMA/EMA)
- Dual HTF (4h + 12h) provides stronger trend confirmation than single HTF
- RSI 45-55 zone (not 40-60) for tighter pullback entries
- Volume spike filter reduces false breakouts
- ADX regime filter avoids choppy markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_dualhtf_volume_1h_4h_12h_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise using Efficiency Ratio (ER)
    
    ER = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    SC = [ER * (fast_SC - slow_SC) + slow_SC]^2
    KAMA = KAMA_prev + SC * (Close - KAMA_prev)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i - 1]
            continue
        
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Calculate ATR for TR
    atr = calculate_atr(high, low, close, period)
    
    # Calculate +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * plus_dm_s[i] / atr[i]
            minus_di[i] = 100 * minus_dm_s[i] / atr[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    return adx


def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes (> threshold * average volume)"""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    spike = volume > (threshold * vol_avg)
    return spike


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF KAMA indicators
    kama_4h = calculate_kama(df_4h['close'].values, period=21, fast_period=2, slow_period=30)
    kama_12h = calculate_kama(df_12h['close'].values, period=50, fast_period=2, slow_period=30)
    
    # Align HTF indicators to LTF (auto shift(1) for completed bars)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Calculate 1h indicators
    kama_1h = calculate_kama(close, period=21, fast_period=2, slow_period=30)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    volume_spike = calculate_volume_spike(volume, 20, 1.5)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.25  # Base position size (25% of capital)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(kama_4h_aligned[i]) or np.isnan(kama_12h_aligned[i]) or
            np.isnan(kama_1h[i]) or np.isnan(atr[i]) or np.isnan(rsi[i]) or 
            np.isnan(adx[i]) or atr[i] == 0):
            signals[i] = 0.0
            continue
        
        # HTF trend filters (both 4h and 12h must align)
        htf_4h_trend = 1 if close[i] > kama_4h_aligned[i] else -1
        htf_12h_trend = 1 if close[i] > kama_12h_aligned[i] else -1
        
        # 1h KAMA trend
        ltf_trend = 1 if close[i] > kama_1h[i] else -1
        
        # ADX regime filter (trending market only)
        regime_valid = adx[i] > 25
        
        # RSI pullback zone (45-55 for entry timing - tighter than 40-60)
        rsi_pullback_long = 45 <= rsi[i] <= 55
        rsi_pullback_short = 45 <= rsi[i] <= 55
        
        # Volume confirmation (only required for new entries)
        vol_confirmed = volume_spike[i]
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: All HTF bullish + LTF bullish + RSI pullback + ADX valid + Volume
        if (htf_4h_trend == 1 and htf_12h_trend == 1 and ltf_trend == 1 and 
            rsi_pullback_long and regime_valid and vol_confirmed):
            target_signal = SIZE
        
        # Short entry: All HTF bearish + LTF bearish + RSI pullback + ADX valid + Volume
        elif (htf_4h_trend == -1 and htf_12h_trend == -1 and ltf_trend == -1 and 
              rsi_pullback_short and regime_valid and vol_confirmed):
            target_signal = -SIZE
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.5 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.5*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * atr[i]:  # 2R = 5*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.5 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 5.0 * atr[i]:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            profit_target_hit = False
        elif take_profit_triggered:
            # Reduce position to half at 2R profit
            signals[i] = HALF_SIZE * position_side
            profit_target_hit = True
        else:
            # Apply signal change
            if target_signal != 0.0 and position_side == 0:
                # New entry
                signals[i] = target_signal
                position_side = 1 if target_signal > 0 else -1
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
                entry_price = close[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                if position_side == 1 and ltf_trend == -1:
                    # LTF trend reversed, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                elif position_side == -1 and ltf_trend == 1:
                    # LTF trend reversed, exit short
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = SIZE * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals