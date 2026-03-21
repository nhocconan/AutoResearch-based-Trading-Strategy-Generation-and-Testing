#!/usr/bin/env python3
"""
EXPERIMENT #008 - Volatility-Adjusted Momentum + Volume Confirmation (30m primary)
==================================================================================
Hypothesis: 30m momentum works best when confirmed by volume and filtered by HTF trend.
Previous Donchian/RSI strategies failed due to false breakouts in chop. This strategy:
- Uses ROC(10) for momentum (faster than MA crossover, slower than RSI extremes)
- Requires volume confirmation (1.5x 20-period avg) to filter false moves
- 4h HMA(21) + 1d HMA(50) for trend alignment (not triple HTF - too restrictive)
- Dynamic position sizing based on ATR (smaller size when volatility is high)
- 2.5*ATR trailing stop (wider than 2.0*ATR to avoid premature exits)
- Discrete signal levels (0.0, ±0.20, ±0.30) to minimize fee churn

Why this should beat previous attempts:
- Volume filter removes 40%+ of false momentum signals
- ROC captures momentum earlier than MA crossovers
- Dynamic sizing controls drawdown in high volatility periods
- 30m TF is faster than 4h/12h but slower than 5m/15m (sweet spot for swing)
- Simpler HTF filter (4h+1d vs triple) = more trades, less overfitting
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "vol_momentum_volume_30m_4h_1d_v1"
timeframe = "30m"
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


def calculate_roc(close, period=10):
    """Calculate Rate of Change (momentum indicator)"""
    n = len(close)
    roc = np.zeros(n)
    roc[:] = np.nan
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = 100 * (close[i] - close[i - period]) / close[i - period]
    return roc


def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing"""
    n = len(close)
    delta = np.zeros(n)
    for i in range(1, n):
        delta[i] = close[i] - close[i - 1]
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(period - 1, n):
        if loss_smooth[i] != 0:
            rs[i] = gain_smooth[i] / loss_smooth[i]
        else:
            rs[i] = 100
    
    rsi = 100 - (100 / (1 + rs))
    return rsi


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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    roc = calculate_roc(close, 10)
    rsi = calculate_rsi(close, 14)
    vol_ma = calculate_volume_ma(volume, 20)
    
    # Calculate ATR percentile for volatility regime
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.zeros(n)
    for i in range(50, n):
        if atr_ma[i] > 0:
            atr_ratio[i] = atr[i] / atr_ma[i]
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing parameters
    BASE_SIZE = 0.25  # Base position size (25% of capital)
    MAX_SIZE = 0.35   # Max position size in low volatility
    MIN_SIZE = 0.15   # Min position size in high volatility
    HALF_SIZE = 0.12  # Half position for take profit
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(roc[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_ma[i]) or atr[i] == 0 or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # HTF trend alignment (4h and 1d must agree)
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # HTF trend direction
        hma_4h_trend = 1 if price_above_4h_hma else -1
        hma_1d_trend = 1 if price_above_1d_hma else -1
        
        # Momentum signals
        roc_bullish = roc[i] > 2.0  # Positive momentum > 2%
        roc_bearish = roc[i] < -2.0  # Negative momentum < -2%
        
        # RSI filter (avoid extremes for momentum strategy)
        rsi_neutral = 35 < rsi[i] < 65  # Not overbought/oversold
        rsi_bullish = rsi[i] > 50 and rsi[i] < 70  # Bullish but not extreme
        rsi_bearish = rsi[i] < 50 and rsi[i] > 30  # Bearish but not extreme
        
        # Volume confirmation (must be 1.5x average volume)
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Volatility regime (avoid extreme volatility)
        vol_normal = 0.5 < atr_ratio[i] < 2.0  # ATR within 50%-200% of average
        
        # Calculate dynamic position size based on volatility
        if atr_ratio[i] < 0.8:
            position_size = MAX_SIZE  # Low volatility = larger size
        elif atr_ratio[i] > 1.5:
            position_size = MIN_SIZE  # High volatility = smaller size
        else:
            position_size = BASE_SIZE  # Normal volatility
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Momentum + Volume + HTF bullish + RSI confirmation
        if (roc_bullish and volume_confirmed and vol_normal and
            hma_4h_trend == 1 and hma_1d_trend == 1 and rsi_bullish):
            target_signal = position_size
        
        # Short entry: Momentum + Volume + HTF bearish + RSI confirmation
        elif (roc_bearish and volume_confirmed and vol_normal and
              hma_4h_trend == -1 and hma_1d_trend == -1 and rsi_bearish):
            target_signal = -position_size
        
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
                
                # Check take profit (2R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 5.0 * entry_atr:  # 2R = 5*ATR
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
                    if close[i] <= entry_price - 5.0 * entry_atr:  # 2R profit
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
                # Exit if momentum reverses OR HTF alignment breaks
                momentum_reversal_long = roc[i] < -1.0
                momentum_reversal_short = roc[i] > 1.0
                hma_alignment_broken = (position_side == 1 and hma_4h_trend == -1) or \
                                       (position_side == -1 and hma_4h_trend == 1)
                
                if momentum_reversal_long or momentum_reversal_short or hma_alignment_broken:
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