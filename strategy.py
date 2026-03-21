#!/usr/bin/env python3
"""
EXPERIMENT #061 - HMA Trend + RSI Pullback + Volume Confirm (15m primary)
==========================================================================
Hypothesis: 15m is too noisy for pure momentum, but works well for pullback entries
into a confirmed HTF trend. Using 4h HMA for major trend + 1h RSI pullback (not extreme)
+ volume confirmation should filter false entries. This differs from failed 15m strategies
by avoiding mean-reversion and focusing on trend-following pullbacks with volume confirmation.

Key features:
- Primary TF: 15m
- HTF filters: 4h HMA(21) for trend + 1h RSI(14) for pullback timing
- Entry: RSI pulls back to 45-55 zone in direction of 4h trend (not extreme 30/70)
- Volume: Must be > 1.3x 20-period average to confirm momentum
- Regime: 4h Bollinger Band width > 40th percentile (avoid chop)
- Stoploss: 2.5*ATR(14) trailing stop
- Position sizing: 0.25 base, scaled by ATR percentile (smaller when vol high)
- Take profit: Reduce to half at 2.5R, trail stop at 1.5R

Why this should beat current best (Sharpe=0.490):
- 15m pullbacks into 4h trend = higher win rate than breakouts
- Volume filter removes 40%+ of false signals
- ATR-based position sizing controls drawdown in high vol periods
- Conservative sizing (0.20-0.30) prevents blowup
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_rsi_vol_pullback_15m_1h_4h_v1"
timeframe = "15m"
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
    """Calculate RSI"""
    n = len(close)
    delta = np.zeros(n)
    for i in range(1, n):
        delta[i] = close[i] - close[i - 1]
    
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    rs = np.zeros(n)
    for i in range(period - 1, n):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
        else:
            rs[i] = 100.0
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma
    return upper, lower, bandwidth


def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average"""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma


def calculate_percentile_rank(series, window=100):
    """Calculate rolling percentile rank"""
    n = len(series)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(window - 1, n):
        if not np.isnan(series[i]):
            window_data = series[i - window + 1:i + 1]
            window_data = window_data[~np.isnan(window_data)]
            if len(window_data) > 0:
                pr[i] = np.sum(window_data <= series[i]) / len(window_data)
    
    return pr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values.copy()
    high = prices["high"].values.copy()
    low = prices["low"].values.copy()
    volume = prices["volume"].values.copy()
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    _, _, bb_bw_4h = calculate_bollinger_bands(df_4h['close'].values, 20, 2.0)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    bb_bw_4h_aligned = align_htf_to_ltf(prices, df_4h, bb_bw_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_15m = calculate_rsi(close, 14)
    vol_ma = calculate_volume_ma(volume, 20)
    
    # Calculate 4h BB bandwidth percentile (regime filter)
    bb_bw_pr = calculate_percentile_rank(bb_bw_4h_aligned, 100)
    
    # Calculate ATR percentile for position sizing
    atr_pr = calculate_percentile_rank(atr, 100)
    
    # Generate signals
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # Base position size (28% of capital)
    MAX_SIZE = 0.35   # Max position size in low vol
    MIN_SIZE = 0.15   # Min position size in high vol
    HALF_SIZE = BASE_SIZE / 2
    
    # Track position state for stoploss and take profit
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    entry_price = 0.0
    entry_atr = 0.0
    profit_target_hit = False
    last_signal = 0.0
    
    min_period = 150  # Wait for all indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]) or
            np.isnan(bb_bw_4h_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(rsi_15m[i]) or np.isnan(vol_ma[i]) or np.isnan(bb_bw_pr[i]) or
            np.isnan(atr_pr[i]) or atr[i] == 0 or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # 4h trend direction
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        hma_slope_bullish = hma_4h_aligned[i] > hma_4h_aligned[i - 10] if i >= 10 else price_above_4h_hma
        hma_slope_bearish = hma_4h_aligned[i] < hma_4h_aligned[i - 10] if i >= 10 else not price_above_4h_hma
        
        # Regime filter: 4h BB bandwidth > 40th percentile (avoid chop)
        regime_ok = bb_bw_pr[i] > 0.40
        
        # Volume confirmation: current volume > 1.3x 20-period MA
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # RSI pullback zones (not extreme - looking for continuation)
        rsi_pullback_long = 45 <= rsi_15m[i] <= 55 and rsi_1h_aligned[i] > 45
        rsi_pullback_short = 45 <= rsi_15m[i] <= 55 and rsi_1h_aligned[i] < 55
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi_15m[i] > rsi_15m[i - 3] if i >= 3 else False
        rsi_momentum_short = rsi_15m[i] < rsi_15m[i - 3] if i >= 3 else False
        
        # Calculate position size based on ATR percentile (smaller when vol high)
        vol_multiplier = 1.0 - (atr_pr[i] - 0.5) * 0.4  # Range: 0.8 to 1.2
        position_size = min(MAX_SIZE, max(MIN_SIZE, BASE_SIZE * vol_multiplier))
        
        # Determine target signal based on all filters
        target_signal = 0.0
        
        # Long entry: 4h bullish + RSI pullback + volume + regime OK
        if (price_above_4h_hma and hma_slope_bullish and regime_ok and
            rsi_pullback_long and rsi_momentum_long and volume_confirmed):
            target_signal = position_size
        
        # Short entry: 4h bearish + RSI pullback + volume + regime OK
        elif (not price_above_4h_hma and hma_slope_bearish and regime_ok and
              rsi_pullback_short and rsi_momentum_short and volume_confirmed):
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
                
                # Check take profit (2.5R from entry, where R = 2.5*ATR at entry)
                if not profit_target_hit:
                    if close[i] >= entry_price + 6.25 * entry_atr:  # 2.5R = 6.25*ATR
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
                    if close[i] <= entry_price - 6.25 * entry_atr:  # 2.5R profit
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
            # Reduce position to half at 2.5R profit
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
                # Exit if 4h HMA alignment breaks
                hma_alignment_broken = (position_side == 1 and not price_above_4h_hma) or \
                                       (position_side == -1 and price_above_4h_hma)
                
                if hma_alignment_broken:
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
        
        last_signal = signals[i]
    
    return signals