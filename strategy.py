#!/usr/bin/env python3
"""
Experiment #033: 1d Donchian Breakout + 1w HMA Trend + ATR Risk Management

Hypothesis: 1d primary timeframe with 1w HTF trend filter will capture major crypto moves
while avoiding counter-trend trades during bear markets. This builds on proven patterns:
- Donchian breakout + HMA trend + RSI + ATR (SOL Sharpe +0.782 in research)
- 1w HMA(21) for major trend direction (call ONCE before loop via mtf_data)
- 1d Donchian(20) breakout for entry signals
- RSI(14) > 50 for longs, < 50 for shorts (simple filter to ensure trades)
- ATR(14) for stoploss (2.5x) and trailing
- Discrete sizing: 0.25 base, 0.30 strong trend alignment

Why this should work:
- 1w HMA filter prevents counter-trend trades (critical in 2022 crash)
- 1d Donchian breakouts capture momentum moves without over-filtering
- Simple RSI filter (>50/<50) ensures trades actually trigger (learned from 0-trade failures)
- 1d TF targets 20-50 trades/year (optimal for fee efficiency on daily)
- ATR stoploss protects from major drawdowns

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_1w_hma_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    loss_avg = np.where(loss_avg == 0, 1e-10, loss_avg)
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA trend
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    sma_200 = calculate_sma(close, 200)
    
    # Also calculate 1d HMA for additional trend confirmation
    hma_1d_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    WEAK_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1d_21[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(sma_200[i]):
            continue
        
        # === 1W HTF TREND BIAS ===
        # Simple: price above 1w HMA = bullish bias, below = bearish
        htf_bullish = close[i] > hma_1w_21_aligned[i]
        htf_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === 1D LOCAL TREND ===
        local_bullish = close[i] > hma_1d_21[i]
        local_bearish = close[i] < hma_1d_21[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI FILTER (simple, not over-filtered) ===
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        
        # === DONCHIAN BREAKOUT ===
        # Long: price breaks above Donchian upper
        # Short: price breaks below Donchian lower
        donchian_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === POSITION SIZING BASED ON TREND STRENGTH ===
        # Strong trend (HTF + local + SMA200 agree) = 0.30
        # Medium trend (HTF + local agree) = 0.25
        # Weak trend (only HTF) = 0.20
        if htf_bullish and local_bullish and above_sma200:
            current_size = STRONG_SIZE
        elif htf_bullish and local_bullish:
            current_size = BASE_SIZE
        elif htf_bullish:
            current_size = WEAK_SIZE
        elif htf_bearish and local_bearish and below_sma200:
            current_size = STRONG_SIZE
        elif htf_bearish and local_bearish:
            current_size = BASE_SIZE
        elif htf_bearish:
            current_size = WEAK_SIZE
        else:
            current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: 1w bullish + Donchian breakout + RSI > 50
        # Relaxed: only need 2 of 3 conditions for entry to ensure trades
        long_conditions = sum([htf_bullish, donchian_long, rsi_bullish])
        if long_conditions >= 2 and htf_bullish:
            new_signal = current_size
        
        # SHORT ENTRY: 1w bearish + Donchian breakout + RSI < 50
        short_conditions = sum([htf_bearish, donchian_short, rsi_bearish])
        if short_conditions >= 2 and htf_bearish:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~60 days on 1d), allow weaker entry
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if htf_bullish and donchian_long:
                new_signal = current_size * 0.8
            elif htf_bearish and donchian_short:
                new_signal = -current_size * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1w trend turns bearish
            if position_side > 0 and htf_bearish:
                trend_reversal = True
            # Exit short if 1w trend turns bullish
            if position_side < 0 and htf_bullish:
                trend_reversal = True
        
        # === RSI EXTREME EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long when RSI becomes very overbought
            if position_side > 0 and rsi_14[i] > 75:
                rsi_exit = True
            # Exit short when RSI becomes very oversold
            if position_side < 0 and rsi_14[i] < 25:
                rsi_exit = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal or rsi_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals