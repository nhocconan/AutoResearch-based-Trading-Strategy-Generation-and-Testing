#!/usr/bin/env python3
"""
EXPERIMENT #063 - HMA Trend + RSI Pullback + Z-Score Filter (1h primary)
=====================================================================================
Hypothesis: 1h timeframe captures swing trades better than 12h for crypto volatility.
4h HMA provides trend direction, 1h RSI identifies pullback entries within the trend.
Z-score(20) filter ensures we enter at mean-reversion points, not chasing extremes.
Volume confirmation adds conviction. This differs from Donchian by using pullback
entries (buying dips in uptrend) rather than breakouts.

Key features:
- Primary TF: 1h
- HTF filter: 4h HMA(21) for trend direction
- Trend: Price vs 4h HMA alignment
- Entry: RSI(14) pullback (30-40 long, 60-70 short) + Z-score confirmation
- Regime: Z-score within ±2 std (avoid extremes)
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit

Why this should beat current best (Sharpe=0.490):
- 1h captures more swing opportunities than 12h
- Pullback entries have better risk/reward than breakouts
- Z-score filter avoids chasing overextended moves
- Conservative sizing controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_rsi_zscore_pullback_1h_4h_v1"
timeframe = "1h"
leverage = 1.0


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
    """Calculate RSI (Relative Strength Index)"""
    n = len(close)
    delta = np.diff(close, prepend=close[0])
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100
    
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100  # No losses = RSI 100
    
    return rsi


def calculate_zscore(close, period=20):
    """Calculate Z-score (standardized deviation from mean)"""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    
    zscore = (close_s - rolling_mean) / rolling_std
    zscore = zscore.fillna(0).values
    
    return zscore


def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average"""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for trend direction
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    zscore = calculate_zscore(close, 20)
    vol_ma = calculate_volume_ma(volume, 20)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.35   # Max position size
    MIN_SIZE = 0.20   # Min position size
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    profit_target_hit = False
    entry_atr = 0.0
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(rsi[i]) or np.isnan(zscore[i]) or np.isnan(vol_ma[i]) or
            atr[i] == 0 or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend direction
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        hma_trend = 1 if price_above_4h_hma else -1
        
        # Volume confirmation (volume > 1.2x average)
        volume_confirmed = volume[i] > 1.2 * vol_ma[i]
        
        # Z-score regime filter (avoid extreme overbought/oversold)
        zscore_normal = abs(zscore[i]) < 2.0
        
        # RSI pullback signals
        # Long: RSI pulled back to 30-45 in uptrend (buying the dip)
        rsi_pullback_long = 30 <= rsi[i] <= 45
        
        # Short: RSI pulled back to 55-70 in downtrend (selling the rally)
        rsi_pullback_short = 55 <= rsi[i] <= 70
        
        # Calculate position size based on Z-score conviction
        zscore_conviction = 1.0 - abs(zscore[i]) / 3.0  # Higher conviction near mean
        zscore_conviction = max(0.7, min(1.0, zscore_conviction))
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * zscore_conviction))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: 4h uptrend + RSI pullback + Z-score normal + volume confirmation
        if (hma_trend == 1 and rsi_pullback_long and zscore_normal and volume_confirmed):
            target_signal = position_size
        
        # Short entry: 4h downtrend + RSI pullback + Z-score normal + volume confirmation
        elif (hma_trend == -1 and rsi_pullback_short and zscore_normal and volume_confirmed):
            target_signal = -position_size
        
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
                
                # Check take profit (2R from entry, where R = 2*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * entry_atr:  # 2R = 4*ATR
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
                    if close[i] <= entry_price - 4.0 * entry_atr:  # 2R profit
                        take_profit_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
            entry_price = 0.0
            entry_atr = 0.0
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
                entry_atr = atr[i]
                profit_target_hit = False
            elif position_side != 0:
                # Maintain existing position (check if trend reversed)
                # Exit if 4h HMA trend flips against position
                hma_trend_reversed = (position_side == 1 and hma_trend == -1) or \
                                     (position_side == -1 and hma_trend == 1)
                
                # Also exit if RSI becomes extreme against position
                rsi_extreme_long = position_side == 1 and rsi[i] > 75
                rsi_extreme_short = position_side == -1 and rsi[i] < 25
                
                if hma_trend_reversed or rsi_extreme_long or rsi_extreme_short:
                    signals[i] = 0.0
                    position_side = 0
                    highest_since_entry = 0.0
                    lowest_since_entry = float('inf')
                    entry_price = 0.0
                    entry_atr = 0.0
                    profit_target_hit = False
                else:
                    # Maintain position
                    signals[i] = position_size * position_side if not profit_target_hit else HALF_SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals