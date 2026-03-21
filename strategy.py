#!/usr/bin/env python3
"""
EXPERIMENT #035 - KAMA Adaptive Trend + RSI Pullback + Volume Filter (12h primary, 1d HTF)
==========================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility - fast 
during trends, slow during chop. Combined with 1d HMA trend filter and RSI pullback 
entries, this should capture sustained trends while avoiding whipsaws. Volume filter 
confirms genuine breakouts. 12h timeframe balances signal frequency with trend quality.

Key features:
- Primary TF: 12h (captures medium-term trends)
- HTF filter: 1d HMA(50) for major trend direction
- Trend: KAMA(10,2,30) adaptive moving average
- Entry: RSI(14) pullback to 45-55 zone in trend direction
- Volume filter: volume > 1.2 * 20-bar MA volume (confirms moves)
- Regime: ADX(14) > 25 (trending market only)
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "kama_rsi_volume_12h_1d_v1"
timeframe = "12h"
leverage = 1.0


def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average
    KAMA adapts to market noise - fast during trends, slow during chop
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio (ER)
    change = close_s.diff(er_period).abs()
    volatility = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    er = change / (volatility + 1e-10)
    er = er.fillna(0)
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i - 1]
        else:
            kama[i] = kama[i - 1] + sc.iloc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


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
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i - 1]) if (high[i] - high[i - 1]) > (low[i - 1] - low[i]) else 0
        minus_dm[i] = max(0, low[i - 1] - low[i]) if (low[i - 1] - low[i]) > (high[i] - high[i - 1]) else 0
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    
    tr_s = pd.Series(tr)
    plus_dm_s = pd.Series(plus_dm)
    minus_dm_s = pd.Series(minus_dm)
    
    atr = tr_s.ewm(span=period, adjust=False, min_periods=period).mean()
    plus_di = 100 * (plus_dm_s.ewm(span=period, adjust=False, min_periods=period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm_s.ewm(span=period, adjust=False, min_periods=period).mean() / (atr + 1e-10))
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False, min_periods=period).mean()
    
    return adx.values, plus_di.values, minus_di.values


def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average"""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean()
    return vol_ma.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    vol_ma = calculate_volume_ma(volume, 20)
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.28  # Base position size (28% of capital)
    HALF_SIZE = SIZE / 2  # For take profit reduction
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_ma[i]) or atr[i] == 0 or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Daily trend filter (HTF) - price above/below 1d HMA(50)
        daily_trend = 1 if close[i] > hma_1d_aligned[i] else -1
        
        # 12h KAMA trend direction
        kama_trend = 1 if close[i] > kama[i] else -1
        
        # ADX regime filter - only trade when ADX > 25 (trending market)
        regime_valid = adx[i] > 25
        
        # Volume filter - volume must be > 1.2 * 20-bar MA
        volume_valid = volume[i] > 1.2 * vol_ma[i]
        
        # RSI pullback zone (45-55 for entry timing - neutral zone)
        rsi_pullback_long = 45 <= rsi[i] <= 55
        rsi_pullback_short = 45 <= rsi[i] <= 55
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: KAMA bullish + Daily trend bullish + RSI pullback + ADX valid + Volume valid
        if kama_trend == 1 and daily_trend == 1 and rsi_pullback_long and regime_valid and volume_valid:
            target_signal = SIZE
        
        # Short entry: KAMA bearish + Daily trend bearish + RSI pullback + ADX valid + Volume valid
        elif kama_trend == -1 and daily_trend == -1 and rsi_pullback_short and regime_valid and volume_valid:
            target_signal = -SIZE
        
        # Stoploss and take profit logic - check BEFORE setting new signal
        stoploss_triggered = False
        take_profit_triggered = False
        
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                
                # Check stoploss
                if close[i] < trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit (2R from entry, where R = 2.0*ATR)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * atr[i]:  # 2R = 4*ATR
                        take_profit_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                
                # Check stoploss
                if close[i] > trailing_stop:
                    stoploss_triggered = True
                
                # Check take profit
                if not profit_target_hit:
                    if close[i] <= entry_price - 4.0 * atr[i]:  # 2R profit
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
                if position_side == 1 and kama_trend == -1:
                    # Trend reversed, exit long
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    profit_target_hit = False
                elif position_side == -1 and kama_trend == 1:
                    # Trend reversed, exit short
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