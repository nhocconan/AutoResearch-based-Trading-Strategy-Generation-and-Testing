#!/usr/bin/env python3
"""
Experiment #028: 30m Primary + 4h HTF — Simplified Trend Pullback

Hypothesis: Previous 30m strategies failed due to TOO MANY confluence filters
(5-6 conditions that never all align = 0 trades). This strategy SIMPLIFIES:

1. 4h HMA(21) for trend direction ONLY (remove 1d - too slow for 30m entries)
2. 30m RSI(14) for pullback entries (simpler than Connors RSI, triggers more)
3. Volume filter (>0.7x avg) - single confirmation
4. NO session filter (crypto trades 24/7, session filter killed trades)
5. ATR(14) trailing stoploss at 2.5x

Why this should work:
- Fewer filters = more trades (target 40-80/year, not 0)
- RSI(14) < 35 / > 65 happens regularly in trends
- 4h trend filter prevents counter-trend trades
- Simpler logic = less chance of mutually exclusive conditions

Timeframe: 30m (REQUIRED)
HTF: 4h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete (0.0, ±0.25)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 40-80/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_pullback_4h_hma_v2"
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

def calculate_slope(series, lookback=5):
    """Calculate linear regression slope over lookback period."""
    n = len(series)
    slope = np.zeros(n)
    for i in range(lookback, n):
        y = series[i-lookback:i+1]
        x = np.arange(lookback)
        if np.std(y) > 0:
            slope[i] = np.polyfit(x, y, 1)[0]
        else:
            slope[i] = 0
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volume SMA for filter
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Price relative to 4h HMA (normalized)
    price_vs_hma = (close - hma_4h_21_aligned) / hma_4h_21_aligned
    
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
        
        if np.isnan(hma_4h_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(volume_sma[i]) or volume_sma[i] == 0:
            continue
        
        # === 4H TREND DIRECTION ===
        # Price above 4h HMA = bullish trend (prefer longs)
        # Price below 4h HMA = bearish trend (prefer shorts)
        # Add buffer to avoid whipsaws at the line
        hma_buffer = 0.005  # 0.5% buffer
        trend_bullish = close[i] > hma_4h_21_aligned[i] * (1 + hma_buffer)
        trend_bearish = close[i] < hma_4h_21_aligned[i] * (1 - hma_buffer)
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.7 * volume_sma[i]
        
        # === RSI PULLBACK ENTRIES ===
        # In bullish trend: wait for RSI pullback to 35-45
        # In bearish trend: wait for RSI rally to 55-65
        rsi_long_entry = rsi_14[i] < 45
        rsi_short_entry = rsi_14[i] > 55
        
        # === EXTREME RSI REVERSAL ===
        # RSI < 25 or > 75 = extreme, can trade against short-term but with HTF trend
        rsi_extreme_low = rsi_14[i] < 25
        rsi_extreme_high = rsi_14[i] > 75
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Primary: Bullish trend + RSI pullback + volume
        if trend_bullish and volume_ok and rsi_long_entry:
            new_signal = current_size
        # Secondary: Extreme oversold in neutral/slightly bullish (RSI < 25)
        elif rsi_extreme_low and volume_ok and close[i] > hma_4h_21_aligned[i] * 0.98:
            new_signal = current_size * 0.6
        
        # SHORT ENTRIES
        # Primary: Bearish trend + RSI rally + volume
        if trend_bearish and volume_ok and rsi_short_entry:
            new_signal = -current_size
        # Secondary: Extreme overbought in neutral/slightly bearish (RSI > 75)
        elif rsi_extreme_high and volume_ok and close[i] < hma_4h_21_aligned[i] * 1.02:
            new_signal = -current_size * 0.6
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 150 bars (~3 days on 30m), allow weaker entry
        if bars_since_last_trade > 150 and new_signal == 0.0 and not in_position:
            if trend_bullish and rsi_14[i] < 50 and volume_ok:
                new_signal = current_size * 0.5
            elif trend_bearish and rsi_14[i] > 50 and volume_ok:
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
        # Exit if HTF trend flips against position
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_bearish and rsi_14[i] > 65:
                trend_reversal = True
            if position_side < 0 and trend_bullish and rsi_14[i] < 35:
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