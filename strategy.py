#!/usr/bin/env python3
"""
Experiment #043: 1d Primary + 1w HTF - Simplified Trend + Mean Reversion

Hypothesis: Daily timeframe reduces noise and fee drag while capturing major trends.
Previous 4h strategies failed due to over-complexity and too many conflicting filters.

This strategy SIMPLIFIES for 1d:
1. 1w HMA(21) for major trend bias (smooth, reduces whipsaws on daily)
2. 1d Choppiness(14) for regime: >61.8=range (mean revert), <38.2=trend (breakout)
3. RSI(14) for entry timing (simpler than Connors, more reliable on 1d)
4. Donchian(20) breakout confirmation for trend entries
5. ATR(14) 2.5x trailing stoploss
6. Discrete position sizing: 0.25 base, max 0.30

Why 1d should beat 4h strategies:
- Less noise = fewer false signals
- Natural 20-50 trades/year (perfect for fee management)
- Better captures major crypto trends (BTC 2021 bull, 2022 bear)
- 1w HTF provides stronger trend filter than 12h

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_rsi_donchian_1w_hma_v1"
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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP > 61.8 = range/choppy market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    atr = calculate_atr(high, low, close, period)
    atr_s = pd.Series(atr)
    atr_sum = atr_s.rolling(window=period, min_periods=period).sum()
    
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    chop = chop.fillna(50).values
    
    return chop

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
    """Calculate Donchian Channel upper and lower bands."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    hma_1d_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1W TREND BIAS ===
        trend_bullish = close[i] > hma_1w_21_aligned[i]
        trend_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINNESS REGIME ===
        choppy_market = chop_14[i] > 61.8
        trending_market = chop_14[i] < 38.2
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_oversold = rsi_14[i] < 25
        rsi_extreme_overbought = rsi_14[i] > 75
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if trend_bullish:
            if choppy_market:
                # Mean reversion in chop: RSI oversold
                if rsi_oversold:
                    new_signal = current_size
            elif trending_market:
                # Trend breakout: Donchian breakout + RSI not overbought
                if donchian_breakout_long and rsi_14[i] < 70:
                    new_signal = current_size
                # Trend pullback: RSI moderate + price > HMA50
                elif rsi_14[i] < 45 and close[i] > hma_1d_50[i]:
                    new_signal = current_size
            else:
                # Neutral regime: RSI extreme oversold
                if rsi_extreme_oversold:
                    new_signal = current_size
        
        # SHORT ENTRIES
        elif trend_bearish:
            if choppy_market:
                # Mean reversion in chop: RSI overbought
                if rsi_overbought:
                    new_signal = -current_size
            elif trending_market:
                # Trend breakout: Donchian breakdown + RSI not oversold
                if donchian_breakout_short and rsi_14[i] > 30:
                    new_signal = -current_size
                # Trend pullback: RSI moderate + price < HMA50
                elif rsi_14[i] > 55 and close[i] < hma_1d_50[i]:
                    new_signal = -current_size
            else:
                # Neutral regime: RSI extreme overbought
                if rsi_extreme_overbought:
                    new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 30 bars (~30 days on 1d), force entry with weaker signal
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if trend_bullish and rsi_14[i] < 40:
                new_signal = current_size * 0.5
            elif trend_bearish and rsi_14[i] > 60:
                new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
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
            if position_side > 0 and trend_bearish:
                trend_reversal = True
            if position_side < 0 and trend_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
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