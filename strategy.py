#!/usr/bin/env python3
"""
EXPERIMENT #004 - HMA Trend + 1d Filter + Volatility Regime (4h primary)
=====================================================================================
Hypothesis: 4h HMA(21) slope captures medium-term trends. 1d HMA(21) filters major trend.
Bollinger Band Width regime detection avoids trading in low-volatility chop.
RSI extremes filter prevents entering at tops/bottoms.

Key features:
- Primary TF: 4h (as required for this experiment)
- HTF filter: 1d HMA(21) for major trend direction
- Trend: 4h HMA(21) slope (rising/falling)
- Regime: Bollinger Band Width percentile (avoid squeeze)
- Entry: RSI not at extremes (30 < RSI < 70)
- Stoploss: 2.0*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
- Take profit: Reduce to half at 2R profit

Why this should work better than failed attempts:
- Simpler than Donchian+Volume (failed #002)
- More robust than EMA+RSI (failed #003)
- Volatility regime filter avoids chop (lesson from KAMA+BB failure)
- 4h timeframe balances signal frequency vs noise
- Conservative sizing controls drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_trend_1dfilter_bbregime_4h_v1"
timeframe = "4h"
leverage = 1.0


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


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period // 2, adjust=False, min_periods=period // 2).mean()
    wma2 = close_s.ewm(span=period, adjust=False, min_periods=period).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False, min_periods=int(np.sqrt(period))).mean()
    return hma.values


def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)"""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_bollinger_bandwidth(close, period=20, std_mult=2.0):
    """Calculate Bollinger Band Width for regime detection"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    return bandwidth, sma, std


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion detection"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for major trend filter
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    hma_21 = calculate_hma(close, 21)
    hma_63 = calculate_hma(close, 63)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    bb_width, bb_sma, bb_std = calculate_bollinger_bandwidth(close, period=20, std_mult=2.0)
    zscore = calculate_zscore(close, period=20)
    
    # Calculate BB Width percentile for regime detection (rolling 100 bars)
    bb_width_percentile = np.zeros(n)
    bb_width_percentile[:] = np.nan
    lookback = 100
    for i in range(lookback, n):
        if not np.isnan(bb_width[i]):
            recent_width = bb_width[i-lookback:i+1]
            recent_width = recent_width[~np.isnan(recent_width)]
            if len(recent_width) > 0:
                bb_width_percentile[i] = np.sum(bb_width[i] >= recent_width) / len(recent_width)
    
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
    entry_atr = 0.0
    profit_target_hit = False
    
    min_period = 100  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(hma_21[i]) or
            np.isnan(hma_63[i]) or np.isnan(atr[i]) or np.isnan(rsi[i]) or
            np.isnan(bb_width[i]) or np.isnan(bb_width_percentile[i]) or
            atr[i] == 0 or bb_std[i] == 0):
            signals[i] = 0.0
            continue
        
        # 1d HMA major trend filter
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        major_trend = 1 if price_above_1d_hma else -1
        
        # 4h HMA trend (slope via HMA21 vs HMA63)
        hma_bullish = hma_21[i] > hma_63[i]
        hma_bearish = hma_21[i] < hma_63[i]
        
        # HMA slope (current vs 5 bars ago)
        hma_slope_long = hma_21[i] > hma_21[i-5] if i >= 5 else False
        hma_slope_short = hma_21[i] < hma_21[i-5] if i >= 5 else False
        
        # Volatility regime (avoid low volatility squeeze)
        regime_expanding = bb_width_percentile[i] > 0.30  # Not in bottom 30%
        
        # RSI filter (not at extremes)
        rsi_ok_long = 35 < rsi[i] < 70  # Not overbought
        rsi_ok_short = 30 < rsi[i] < 65  # Not oversold
        
        # Z-score filter (not too extended)
        zscore_ok_long = zscore[i] < 1.5
        zscore_ok_short = zscore[i] > -1.5
        
        # Calculate position size
        position_size = BASE_SIZE
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: Major trend up + HMA bullish + HMA rising + Regime OK + RSI OK
        if (major_trend == 1 and hma_bullish and hma_slope_long and 
            regime_expanding and rsi_ok_long and zscore_ok_long):
            target_signal = position_size
        
        # Short entry: Major trend down + HMA bearish + HMA falling + Regime OK + RSI OK
        elif (major_trend == -1 and hma_bearish and hma_slope_short and 
              regime_expanding and rsi_ok_short and zscore_ok_short):
            target_signal = -position_size
        
        # Stoploss and take profit logic
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
                
                # Check take profit (2R from entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 4.0 * entry_atr:
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
                    if close[i] <= entry_price - 4.0 * entry_atr:
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
                # Maintain existing position or exit on trend reversal
                trend_reversed = (position_side == 1 and not hma_bullish) or \
                                 (position_side == -1 and not hma_bearish)
                major_trend_reversed = (position_side == 1 and major_trend == -1) or \
                                       (position_side == -1 and major_trend == 1)
                
                if trend_reversed or major_trend_reversed:
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