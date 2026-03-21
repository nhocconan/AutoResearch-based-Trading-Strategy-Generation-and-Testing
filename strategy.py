#!/usr/bin/env python3
"""
EXPERIMENT #002 - Multi-Timeframe EMA Trend + RSI Pullback Strategy
====================================================================
Hypothesis: Daily EMA(21/55) trend filter combined with 4h RSI pullback entries
will capture major crypto trends while avoiding chasing extended moves. The daily
trend provides strong directional bias, while 4h RSI < 40 (long) or > 60 (short)
entries on pullbacks give better risk/reward than simple breakouts.

Key features:
- Daily EMA(21) vs EMA(55) for trend direction (HTF filter)
- 4h RSI(14) < 40 for long entries in uptrend (pullback buys)
- 4h RSI(14) > 60 for short entries in downtrend (pullback sells)
- ATR(14) trailing stoploss (2x ATR)
- Discrete position sizing (0.0, ±0.25, ±0.35) - controls drawdown
- Proper MTF alignment using mtf_data helper (NO manual resampling)

Why this should beat baseline:
- Daily trend filter reduces whipsaws significantly vs single-TF
- RSI pullback entries avoid buying tops/selling bottoms
- Conservative sizing (0.35 max) prevents 2022-style drawdowns
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_ema_rsi_pullback_4h_v1"
timeframe = "4h"
leverage = 1.0


def calculate_ema(series, span):
    """Calculate EMA with proper min_periods"""
    return series.ewm(span=span, min_periods=span, adjust=False).mean()


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    n = len(close)
    delta = np.zeros(n)
    delta[1:] = np.diff(close)
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    
    return rsi


def calculate_atr(high, low, close, period=14):
    """Calculate ATR with proper min_periods"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load daily HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMAs
    daily_close = df_daily['close'].values
    ema21_daily = calculate_ema(pd.Series(daily_close), 21).values
    ema55_daily = calculate_ema(pd.Series(daily_close), 55).values
    
    # Align daily data to 4h timeframe (Rule 2 - no manual i//N mapping)
    # align_htf_to_ltf handles shift(1) for completed bars only
    ema21_aligned = align_htf_to_ltf(prices, df_daily, ema21_daily)
    ema55_aligned = align_htf_to_ltf(prices, df_daily, ema55_daily)
    
    # Calculate 4h indicators
    rsi_4h = calculate_rsi(close, 14)
    atr_4h = calculate_atr(high, low, close, 14)
    
    # Generate signals with discrete position sizing (Rule 4)
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25  # Initial entry size
    SIZE_FULL = 0.35   # Full position size (MAX 0.40 per rules)
    
    # Track position state for stoploss (Rule 6)
    position_side = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    # Warmup period for all indicators
    warmup = max(55, 14)  # EMA55 + RSI14
    
    for i in range(warmup, n):
        # Daily trend filter (HTF)
        daily_trend = 0
        if not np.isnan(ema21_aligned[i]) and not np.isnan(ema55_aligned[i]):
            if ema21_aligned[i] > ema55_aligned[i]:
                daily_trend = 1  # Bullish
            elif ema21_aligned[i] < ema55_aligned[i]:
                daily_trend = -1  # Bearish
        
        # RSI pullback signals (LTF)
        rsi_val = rsi_4h[i]
        atr_val = atr_4h[i]
        
        # Long entry: Daily uptrend + RSI pullback < 40
        if daily_trend == 1 and not np.isnan(rsi_val) and rsi_val < 40:
            if position_side <= 0:
                signals[i] = SIZE_ENTRY
                position_side = 1
                entry_price = close[i]
                highest_price = close[i]
        
        # Short entry: Daily downtrend + RSI pullback > 60
        elif daily_trend == -1 and not np.isnan(rsi_val) and rsi_val > 60:
            if position_side >= 0:
                signals[i] = -SIZE_ENTRY
                position_side = -1
                entry_price = close[i]
                lowest_price = close[i]
        
        # Manage long position
        elif position_side == 1:
            highest_price = max(highest_price, close[i])
            
            # Take profit: RSI > 75 (overbought)
            if not np.isnan(rsi_val) and rsi_val > 75:
                signals[i] = 0.0
                position_side = 0
            
            # Stoploss: 2x ATR from entry or trail from high (Rule 6)
            elif atr_val > 0 and (close[i] < entry_price - 2 * atr_val or 
                                  close[i] < highest_price - 2 * atr_val):
                signals[i] = 0.0
                position_side = 0
        
        # Manage short position
        elif position_side == -1:
            lowest_price = min(lowest_price, close[i])
            
            # Take profit: RSI < 25 (oversold)
            if not np.isnan(rsi_val) and rsi_val < 25:
                signals[i] = 0.0
                position_side = 0
            
            # Stoploss: 2x ATR from entry or trail from low (Rule 6)
            elif atr_val > 0 and (close[i] > entry_price + 2 * atr_val or 
                                  close[i] > lowest_price + 2 * atr_val):
                signals[i] = 0.0
                position_side = 0
        
        # Trend reversal exit
        if position_side == 1 and daily_trend == -1:
            signals[i] = 0.0
            position_side = 0
        elif position_side == -1 and daily_trend == 1:
            signals[i] = 0.0
            position_side = 0
    
    return signals