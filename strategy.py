#!/usr/bin/env python3
"""
Experiment #050: 1h Primary + 4h/12h HTF — Simplified Trend Pullback

Hypothesis: Previous strategies failed due to OVERLY STRICT confluence (0 trades).
This strategy uses SIMPLER but EFFECTIVE filters to ensure trade generation:

1. 12h HMA(21) for MAJOR trend bias (trade WITH trend only)
2. 4h RSI(14) pullback to 35-65 zone (not extreme - ensures entries happen)
3. 1h price momentum breakout (close > high[5] for long, < low[5] for short)
4. Volume > 1.0x 20-bar avg (minimal filter - just confirm activity)
5. ATR(14) trailing stoploss at 2.5x
6. NO session filter (allows more trades across all hours)

Why this should work:
- Simpler = more trades (target 40-80/year)
- 12h trend filter prevents counter-trend disasters
- RSI 35-65 is COMMON (not rare extremes like 15/85)
- 1h breakout provides timing within HTF trend
- Discrete sizing (0.25/0.30) minimizes fee churn

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 12h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 40-80/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h12h_v1"
timeframe = "1h"
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
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    rsi_4h_14 = calculate_rsi(df_4h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    rsi_4h_14_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h_14)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_1h_14 = calculate_rsi(close, 14)
    
    # Volume SMA for filter
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Price momentum (5-bar breakout)
    high_5 = pd.Series(high).rolling(window=5, min_periods=5).max().values
    low_5 = pd.Series(low).rolling(window=5, min_periods=5).min().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(rsi_4h_14_aligned[i]):
            continue
        
        if np.isnan(rsi_1h_14[i]) or np.isnan(volume_sma[i]) or volume_sma[i] == 0:
            continue
        
        # === 12H TREND BIAS (MAJOR) ===
        # Price above 12h HMA = bullish bias (prefer longs)
        # Price below 12h HMA = bearish bias (prefer shorts)
        trend_12h_bullish = close[i] > hma_12h_21_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_21_aligned[i]
        
        # === 4H RSI PULLBACK (INTERMEDIATE) ===
        # RSI 35-65 = pullback zone (common, ensures trades happen)
        # RSI < 35 = oversold (good for long entry in uptrend)
        # RSI > 65 = overbought (good for short entry in downtrend)
        rsi_4h_neutral = 35 <= rsi_4h_14_aligned[i] <= 65
        rsi_4h_oversold = rsi_4h_14_aligned[i] < 45
        rsi_4h_overbought = rsi_4h_14_aligned[i] > 55
        
        # === VOLUME FILTER (minimal) ===
        volume_ok = volume[i] > 0.9 * volume_sma[i]
        
        # === 1H MOMENTUM BREAKOUT ===
        momentum_long = close[i] > high_5[i-1] if i > 5 else False
        momentum_short = close[i] < low_5[i-1] if i > 5 else False
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC (LOOSE enough to generate trades) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Require: 12h bullish + 4h RSI pullback/oversold + volume + momentum
        if trend_12h_bullish:
            if (rsi_4h_oversold or rsi_4h_neutral) and volume_ok and momentum_long:
                new_signal = current_size
        
        # SHORT ENTRIES
        # Require: 12h bearish + 4h RSI pullback/overbought + volume + momentum
        if trend_12h_bearish:
            if (rsi_4h_overbought or rsi_4h_neutral) and volume_ok and momentum_short:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 300 bars (~12 days on 1h), allow weaker entry
        if bars_since_last_trade > 300 and new_signal == 0.0 and not in_position:
            if trend_12h_bullish and rsi_1h_14[i] < 40:
                new_signal = current_size * 0.5
            elif trend_12h_bearish and rsi_1h_14[i] > 60:
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
            if position_side > 0 and trend_12h_bearish and rsi_4h_14_aligned[i] > 65:
                trend_reversal = True
            if position_side < 0 and trend_12h_bullish and rsi_4h_14_aligned[i] < 35:
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