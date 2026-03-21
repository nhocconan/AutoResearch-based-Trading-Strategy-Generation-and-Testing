#!/usr/bin/env python3
"""
Experiment #044: 30m Fisher Transform + 4h/1d HMA Regime Strategy
Hypothesis: 30m timeframe with Ehlers Fisher Transform for entry timing captures
reversals better than RSI in bear/range markets. 4h HMA provides trend filter,
1d HMA provides macro regime. Fisher crosses at extremes (-1.5/+1.5) give clear
entry signals with fewer whipsaws than RSI. Conservative sizing (0.25) with
2.5x ATR stops controls drawdown. Target: 40-60 trades/year to balance fee drag
vs opportunity capture.

Key innovations vs failed 30m strategies:
- Fisher Transform instead of RSI (better reversal detection)
- Fewer entry conditions (avoid over-filtering = 0 trades)
- Strong HTF regime filter (4h + 1d HMA) prevents counter-trend trades
- Discrete signal levels minimize churn costs
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_4h_1d_hma_regime_v1"
timeframe = "30m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Crosses at extremes (-1.5, +1.5) signal reversals.
    Reference: Ehlers, J.F. "Cycle Analytics for Traders"
    """
    hl2 = (high + low) / 2
    # Normalize price to -1 to +1 range
    highest = pd.Series(hl2).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(hl2).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val = np.where(range_val < 0.0001, 0.0001, range_val)
    
    normalized = 2 * ((hl2 - lowest) / range_val) - 1
    normalized = np.clip(normalized, -0.999, 0.999)  # Fisher domain
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    fisher = np.nan_to_num(fisher, nan=0.0)
    
    # Signal line (previous fisher)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
    supertrend[0] = lower[0]
    direction[0] = 1
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = lower[i]
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            supertrend[i] = upper[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
    
    return supertrend, direction

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    rsi = calculate_rsi(close, 14)
    
    # 30m HMA for trend
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    SIZE = 0.25
    HALF_SIZE = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    max_profit = 0.0
    
    for i in range(100, n):
        # 1d macro regime (bull/bear)
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        
        # 4h trend filter
        hma4h_bullish = hma_4h_aligned[i] > 0 and close[i] > hma_4h_aligned[i]
        hma4h_bearish = hma_4h_aligned[i] > 0 and close[i] < hma_4h_aligned[i]
        
        # 30m Supertrend direction
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Supertrend flip signals
        st_flip_long = st_direction[i] == 1 and st_direction[i-1] == -1
        st_flip_short = st_direction[i] == -1 and st_direction[i-1] == 1
        
        # 30m HMA trend
        hma_trend_long = hma_21[i] > hma_50[i]
        hma_trend_short = hma_21[i] < hma_50[i]
        
        # Fisher Transform signals (Ehlers method)
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        fisher_cross_up = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_cross_down = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        
        # Fisher extreme levels
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # RSI confirmation (relaxed thresholds)
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # Price position
        price_above_hma21 = close[i] > hma_21[i]
        price_below_hma21 = close[i] < hma_21[i]
        
        new_signal = 0.0
        
        # LONG ENTRY TRIGGERS (multiple paths for sufficient trades)
        # Trigger 1: Supertrend flip long + daily bullish (strongest)
        if st_flip_long and daily_bullish:
            new_signal = SIZE
        # Trigger 2: Fisher cross up + 4h bullish + Supertrend long
        elif fisher_cross_up and hma4h_bullish and st_long:
            new_signal = SIZE
        # Trigger 3: Fisher oversold + daily bullish + price above HMA21 (pullback)
        elif fisher_oversold and daily_bullish and price_above_hma21:
            new_signal = SIZE
        # Trigger 4: Supertrend long + HMA trend long + RSI oversold (trend continuation)
        elif st_long and hma_trend_long and rsi_oversold:
            new_signal = SIZE
        # Trigger 5: 4h bullish + Supertrend flip long (HTF confirmation)
        elif hma4h_bullish and st_flip_long:
            new_signal = SIZE
        
        # SHORT ENTRY TRIGGERS
        # Trigger 1: Supertrend flip short + daily bearish (strongest)
        if st_flip_short and daily_bearish:
            new_signal = -SIZE
        # Trigger 2: Fisher cross down + 4h bearish + Supertrend short
        elif fisher_cross_down and hma4h_bearish and st_short:
            new_signal = -SIZE
        # Trigger 3: Fisher overbought + daily bearish + price below HMA21 (rally fade)
        elif fisher_overbought and daily_bearish and price_below_hma21:
            new_signal = -SIZE
        # Trigger 4: Supertrend short + HMA trend short + RSI overbought (trend continuation)
        elif st_short and hma_trend_short and rsi_overbought:
            new_signal = -SIZE
        # Trigger 5: 4h bearish + Supertrend flip short (HTF confirmation)
        elif hma4h_bearish and st_flip_short:
            new_signal = -SIZE
        
        # Stoploss and take profit logic (Rule 6)
        if position_side > 0 and entry_price > 0:
            # Initial stoploss
            stop_loss = entry_price - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for longs
                new_trailing = close[i] - 2.5 * atr[i]
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                if close[i] < trailing_stop and trailing_stop > entry_price:
                    new_signal = 0.0
                # Track max profit for take profit
                if close[i] > entry_price:
                    max_profit = max(max_profit, close[i] - entry_price)
                # Take partial profit at 2.5R
                if max_profit >= 2.5 * atr[i] and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price > 0:
            # Initial stoploss
            stop_loss = entry_price + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            else:
                # Trail stop for shorts
                new_trailing = close[i] + 2.5 * atr[i]
                if new_trailing < trailing_stop or trailing_stop == 0:
                    trailing_stop = new_trailing
                if close[i] > trailing_stop and trailing_stop < entry_price:
                    new_signal = 0.0
                # Track max profit for take profit
                if close[i] < entry_price:
                    max_profit = max(max_profit, entry_price - close[i])
                # Take partial profit at 2.5R
                if max_profit >= 2.5 * atr[i] and signals[i-1] == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            max_profit = 0.0
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price = close[i]
                position_side = np.sign(new_signal)
                trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
                max_profit = 0.0
        elif new_signal == 0 and position_side != 0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            max_profit = 0.0
        
        signals[i] = new_signal
    
    return signals