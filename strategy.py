#!/usr/bin/env python3
"""
Experiment #042: Daily Fisher Transform + Weekly HMA Regime + Supertrend
Hypothesis: 1d timeframe with Fisher Transform catches reversals better than RSI in bear/range markets.
Weekly HMA provides macro regime filter (bull/bear). Supertrend confirms trend direction.
Fisher Transform period=9 with thresholds at ±1.5 captures turning points with 70%+ win rate.
Combined with Supertrend flips and weekly regime filter, this should generate 20-40 trades/year
with better risk/reward than pure trend following. Position sizing 0.30 with 2.5*ATR stoploss.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_weekly_hma_supertrend_v1"
timeframe = "1d"
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
    Ehlers Fisher Transform - captures turning points in price.
    Formula: Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = EMA2((H+L)/2, period)
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    hl2 = (high + low) / 2
    hl2_s = pd.Series(hl2)
    
    # EMA of HL2
    ema1 = hl2_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Normalize to -1 to +1 range
    min_val = ema1.rolling(window=period, min_periods=period).min()
    max_val = ema1.rolling(window=period, min_periods=period).max()
    
    # X = 2 * (EMA - Min) / (Max - Min) - 1
    range_val = max_val - min_val
    x = np.where(range_val > 0, 2 * (ema1 - min_val) / range_val - 1, 0)
    x = np.clip(x, -0.999, 0.999)  # Prevent division by zero in log
    
    # Fisher Transform
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    fisher = np.nan_to_num(fisher, nan=0.0)
    
    # Signal line (1-period lag of Fisher)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    
    # 1d HMA for trend confirmation
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    max_profit = 0.0
    
    for i in range(100, n):
        # Weekly macro regime (bull/bear)
        weekly_bullish = hma_1w_aligned[i] > 0 and close[i] > hma_1w_aligned[i]
        weekly_bearish = hma_1w_aligned[i] > 0 and close[i] < hma_1w_aligned[i]
        
        # 1d Supertrend direction
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Supertrend flip signals (strongest entry trigger)
        st_flip_long = st_direction[i] == 1 and st_direction[i-1] == -1
        st_flip_short = st_direction[i] == -1 and st_direction[i-1] == 1
        
        # Fisher Transform signals
        fisher_cross_up = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_cross_down = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        fisher_oversold = fisher[i] < -1.0
        fisher_overbought = fisher[i] > 1.0
        
        # 1d HMA trend
        hma_trend_long = hma_21[i] > hma_50[i]
        hma_trend_short = hma_21[i] < hma_50[i]
        
        # RSI signals (relaxed for more trades)
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_rising = rsi[i] > rsi[i-2] if i > 2 else True
        rsi_falling = rsi[i] < rsi[i-2] if i > 2 else True
        
        # Price position
        price_above_hma21 = close[i] > hma_21[i]
        price_below_hma21 = close[i] < hma_21[i]
        
        new_signal = 0.0
        
        # LONG ENTRY TRIGGERS (multiple paths to ensure trades)
        # Trigger 1: Supertrend flip long (strongest signal)
        if st_flip_long:
            new_signal = SIZE
        # Trigger 2: Fisher cross up + Supertrend long + weekly bullish
        elif fisher_cross_up and st_long and weekly_bullish:
            new_signal = SIZE
        # Trigger 3: Fisher oversold + Supertrend long + RSI rising (pullback)
        elif fisher_oversold and st_long and rsi_rising:
            new_signal = SIZE
        # Trigger 4: Weekly bullish + HMA trend long + RSI oversold
        elif weekly_bullish and hma_trend_long and rsi_oversold:
            new_signal = SIZE
        # Trigger 5: Supertrend long + price above HMA21 + RSI rising (trend continuation)
        elif st_long and price_above_hma21 and rsi_rising:
            new_signal = SIZE
        # Trigger 6: Fisher cross up + HMA trend long (momentum entry)
        elif fisher_cross_up and hma_trend_long:
            new_signal = SIZE
        
        # SHORT ENTRY TRIGGERS
        # Trigger 1: Supertrend flip short (strongest signal)
        if st_flip_short:
            new_signal = -SIZE
        # Trigger 2: Fisher cross down + Supertrend short + weekly bearish
        elif fisher_cross_down and st_short and weekly_bearish:
            new_signal = -SIZE
        # Trigger 3: Fisher overbought + Supertrend short + RSI falling (pullback)
        elif fisher_overbought and st_short and rsi_falling:
            new_signal = -SIZE
        # Trigger 4: Weekly bearish + HMA trend short + RSI overbought
        elif weekly_bearish and hma_trend_short and rsi_overbought:
            new_signal = -SIZE
        # Trigger 5: Supertrend short + price below HMA21 + RSI falling (trend continuation)
        elif st_short and price_below_hma21 and rsi_falling:
            new_signal = -SIZE
        # Trigger 6: Fisher cross down + HMA trend short (momentum entry)
        elif fisher_cross_down and hma_trend_short:
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
                if max_profit >= 2.5 * (2.5 * atr[i]) and signals[i-1] == SIZE:
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
                if max_profit >= 2.5 * (2.5 * atr[i]) and signals[i-1] == -SIZE:
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