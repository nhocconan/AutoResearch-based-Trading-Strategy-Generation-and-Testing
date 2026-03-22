#!/usr/bin/env python3
"""
Experiment #018: 30m HMA Trend + RSI Pullback with 4h/1d Bias

Hypothesis: Previous 30m strategy (#008) failed with 0 trades due to overly
restrictive entry conditions (session + volume + regime filters). This strategy
simplifies to proven patterns from #009 (4h HMA + RSI pullback) but adapts for
30m timeframe with LOOSER entry conditions to ensure trade frequency.

Key changes from failed #008:
- NO session filter (8-20 UTC was too restrictive)
- NO volume filter (caused 0 trades)
- NO Choppiness Index regime (failed in #012, #013)
- SIMPLER: 4h HMA trend + 30m RSI pullback + 1d bias only
- RSI range: 35-65 (not extreme) to catch more entries
- Target: 40-80 trades/year (not 0 like #008)

Why this should work:
- 4h HMA proven in #009 (Sharpe=0.028, only positive strategy)
- 30m entries give better timing within 4h trend
- 1d HMA bias prevents major counter-trend trades
- LOOSE entry conditions ensure trades happen (critical lesson from #008, #010, #017)
- ATR stoploss protects from 2022-style crashes

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h + 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete (smaller for lower TF)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_pullback_4h1d_bias_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    return sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h_16 = calculate_hma(df_4h['close'].values, 16)
    hma_4h_48 = calculate_hma(df_4h['close'].values, 48)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_16_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_16)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    # Lower TF = smaller size to reduce fee impact
    BASE_SIZE = 0.25
    
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
        
        if np.isnan(hma_4h_16_aligned[i]) or np.isnan(hma_4h_48_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(sma_200[i]):
            continue
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H HMA TREND ===
        hma_4h_bullish = hma_4h_16_aligned[i] > hma_4h_48_aligned[i]
        hma_4h_bearish = hma_4h_16_aligned[i] < hma_4h_48_aligned[i]
        
        # === 30M PRICE VS SMA200 ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC (LOOSE CONDITIONS TO ENSURE TRADES) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG: 4h HMA bullish + 1d bias bullish + RSI pullback (35-55 range)
        if hma_4h_bullish and daily_bullish and price_above_sma200:
            # RSI pullback to 35-55 range (cooling off in uptrend)
            if 35 <= rsi_14[i] <= 55:
                new_signal = current_size
            # RSI crossing up from oversold
            elif rsi_14[i] > 30 and i > 0 and rsi_14[i-1] <= 30:
                new_signal = current_size
        
        # SHORT: 4h HMA bearish + 1d bias bearish + RSI pullback (45-65 range)
        elif hma_4h_bearish and daily_bearish and price_below_sma200:
            # RSI pullback to 45-65 range (cooling off in downtrend)
            if 45 <= rsi_14[i] <= 65:
                new_signal = -current_size
            # RSI crossing down from overbought
            elif rsi_14[i] < 70 and i > 0 and rsi_14[i-1] >= 70:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~30 hours on 30m), force entry with weaker signal
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if hma_4h_bullish and daily_bullish and rsi_14[i] > 40:
                new_signal = current_size * 0.5
            elif hma_4h_bearish and daily_bearish and rsi_14[i] < 60:
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
            if position_side > 0 and hma_4h_bearish:
                trend_reversal = True
            if position_side < 0 and hma_4h_bullish:
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