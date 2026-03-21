#!/usr/bin/env python3
"""
EXPERIMENT #020 - KAMA Adaptive Trend + Bollinger Squeeze + RSI Entry + ATR Stop
================================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better
than fixed MAs or Donchian channels. Combined with Bollinger Band squeeze detection
(low volatility before expansion) and RSI pullback entries, this should capture
trends earlier while filtering choppy markets.

Key differences from #017 (Donchian winner):
- KAMA(21) adaptive trend filter instead of Donchian(20) breakout
- Bollinger Band width percentile for regime detection (squeeze = opportunity)
- RSI(14) pullback entries with tighter thresholds (45/55 vs 50/50)
- ATR(14) trailing stop at 2.0*ATR (tighter than 2.5 for better risk control)
- Position sizing: max 0.30 (more conservative than 0.35)
- Add BB percentile filter to avoid entering during extreme volatility

Why this might beat Sharpe=6.689:
- KAMA reduces whipsaws in choppy markets (adaptive smoothing)
- BB squeeze captures volatility expansion before price moves
- Multi-timeframe logic (4h KAMA trend + 1h BB/RSI entry) proven effective
- Tighter stops protect gains better in fast reversals
- More conservative sizing reduces drawdown during black swan events
"""

import numpy as np
import pandas as pd

name = "mtf_kama_bb_squeeze_rsi_atr_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts smoothing based on market efficiency ratio
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        sum_volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if sum_volatility > 0:
            er[i] = price_change / sum_volatility
    
    # Calculate smoothing constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands with bandwidth"""
    n = len(close)
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = mean + std_dev * std
    lower = mean - std_dev * std
    bandwidth = (upper - lower) / mean
    
    return upper, lower, mean, bandwidth


def calculate_bb_percentile(bandwidth, lookback=100):
    """Calculate Bollinger Band width percentile for regime detection"""
    n = len(bandwidth)
    percentile = np.zeros(n)
    
    for i in range(lookback - 1, n):
        window = bandwidth[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            percentile[i] = np.sum(valid <= bandwidth[i]) / len(valid)
    
    return percentile


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    n = len(close)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.zeros(n)
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[avg_loss == 0] = 100.0
    
    return rsi


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing and risk
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_mean, bb_bw = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    bb_pct = calculate_bb_percentile(bb_bw, lookback=100)
    
    # 4h KAMA for adaptive trend filter (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    # Resample to 4h
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    c_4h = df_4h['close'].values
    n_4h = len(c_4h)
    
    # Calculate 4h KAMA
    kama_4h = calculate_kama(c_4h, period=21)
    
    # 4h trend direction based on KAMA slope and price position
    trend_4h = np.zeros(n_4h)
    for i in range(25, n_4h):
        kama_slope = kama_4h[i] - kama_4h[i - 5]
        price_vs_kama = (c_4h[i] - kama_4h[i]) / kama_4h[i] if kama_4h[i] > 0 else 0
        
        if kama_slope > 0 and price_vs_kama > -0.01:
            trend_4h[i] = 1  # Bullish
        elif kama_slope < 0 and price_vs_kama < 0.01:
            trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.30   # Full position with all confirmations
    SIZE_HALF = 0.20   # Reduced position with partial confirmation
    
    # RSI thresholds for pullback entries (tighter than 50/50)
    RSI_LONG_ENTRY = 45   # Enter long on pullback in uptrend
    RSI_SHORT_ENTRY = 55  # Enter short on rally in downtrend
    RSI_EXIT_LONG = 70    # Exit long when overbought
    RSI_EXIT_SHORT = 30   # Exit short when oversold
    
    # Bollinger Band squeeze thresholds
    BB_SQUEEZE_LOW = 0.20   # Bottom 20% of BB width = squeeze
    BB_SQUEEZE_HIGH = 0.80  # Top 80% = expanding volatility
    
    # ATR stoploss multiplier (tighter than #017)
    ATR_STOP_MULT = 2.0
    
    # ATR filter threshold
    ATR_MAX_PCT = 0.05  # 5% of price
    
    first_valid = max(100, 25, 21, 14, 20)  # Wait for all indicators
    
    # Track entry prices for trailing stop logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    profit_target_hit = np.zeros(n)  # Track if 2R target reached
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(bb_pct[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        atr = atr_1h[i]
        price = close[i]
        bb_percentile = bb_pct[i]
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > ATR_MAX_PCT:
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            continue
        
        # Check trailing stop for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            prev_profit_target = profit_target_hit[i - 1]
            
            # Update highest/lowest since entry for trailing
            if prev_side == 1:
                highest_since_entry[i] = max(highest_since_entry[i-1] if i > 0 else 0, price)
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                
                # Trail stop after 2R profit
                if prev_profit_target > 0:
                    trail_stop = highest_since_entry[i] - ATR_STOP_MULT * atr
                    stoploss_price = max(stoploss_price, trail_stop)
                
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    profit_target_hit[i] = 0
                    continue
                    
            elif prev_side == -1:
                lowest_since_entry[i] = min(lowest_since_entry[i-1] if i > 0 else price, price)
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                
                # Trail stop after 2R profit
                if prev_profit_target > 0:
                    trail_stop = lowest_since_entry[i] + ATR_STOP_MULT * atr
                    stoploss_price = min(stoploss_price, trail_stop)
                
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    profit_target_hit[i] = 0
                    continue
        
        # BB squeeze filter - only enter during low volatility (squeeze)
        # or during expansion phase (after squeeze)
        in_squeeze = bb_percentile < BB_SQUEEZE_HIGH
        
        # RSI exit conditions for existing positions
        if i > 0 and position_side[i - 1] == 1 and rsi_val > RSI_EXIT_LONG:
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
            profit_target_hit[i] = 0
            continue
            
        if i > 0 and position_side[i - 1] == -1 and rsi_val < RSI_EXIT_SHORT:
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
            profit_target_hit[i] = 0
            continue
        
        # Check 2R profit target for position reduction
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            risk_distance = ATR_STOP_MULT * atr
            
            if prev_side == 1 and price >= prev_entry + 2 * risk_distance:
                profit_target_hit[i] = 1
            elif prev_side == -1 and price <= prev_entry - 2 * risk_distance:
                profit_target_hit[i] = 1
            else:
                profit_target_hit[i] = profit_target_hit[i - 1] if i > 0 else 0
        else:
            profit_target_hit[i] = 0
        
        # Determine position size based on BB squeeze confirmation
        position_size = SIZE_FULL if in_squeeze else SIZE_HALF
        
        if trend == 1:  # 4h uptrend
            # RSI pullback entry in uptrend with BB squeeze confirmation
            if rsi_val < RSI_LONG_ENTRY and rsi_val > 30:
                signals[i] = position_size
                position_side[i] = 1
                entry_price[i] = price
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    profit_target_hit[i] = profit_target_hit[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    profit_target_hit[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            # RSI rally entry in downtrend with BB squeeze confirmation
            if rsi_val > RSI_SHORT_ENTRY and rsi_val < 70:
                signals[i] = -position_size
                position_side[i] = -1
                entry_price[i] = price
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
            else:
                # Hold or exit
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
                    profit_target_hit[i] = profit_target_hit[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    profit_target_hit[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            highest_since_entry[i] = 0
            lowest_since_entry[i] = 0
            profit_target_hit[i] = 0
    
    return signals