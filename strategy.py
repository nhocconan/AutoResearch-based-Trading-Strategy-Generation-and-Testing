#!/usr/bin/env python3
"""
Experiment #070: 1h Primary + 4h/12h HTF — Trend Pullback Strategy

Hypothesis: Previous 1h strategies failed due to either 0 trades (too strict) or 
too many trades (fee drag). This strategy uses PROVEN pattern:

1. 4h HMA(21) for MAJOR trend direction (price vs HMA, not slope)
2. 12h ADX(14) for regime filter (>20 = trending, <20 = reduce size)
3. 1h RSI(14) pullback entries (40-60 range, NOT extremes 20/80)
4. Volume filter: >0.6x 20-bar avg (lenient to ensure trades)
5. NO session filter (1h already limits frequency)
6. ATR(14) stoploss at 2.5x with trailing
7. Position size: 0.25 discrete (conservative for 1h)

Why this should work on 1h:
- 4h trend filter ensures we trade WITH higher TF momentum
- RSI 40-60 captures pullbacks in trends (more frequent than 20/80 extremes)
- ADX regime filter reduces size in choppy markets
- Volume filter prevents low-liquidity entries but is lenient (0.6x)
- Target: 40-80 trades/year per symbol (160-320 over 4 years train)

CRITICAL: Entry conditions must be PERMISSIVE enough to generate trades.
If RSI threshold is 35-45, that's 10% of bars. With 4h trend filter (~50% of time),
we get ~5% of bars as potential entries = ~1750 potential entries over 35K bars.
With ADX and volume filters, should get 200-400 actual trades over 4 years.

Timeframe: 1h (REQUIRED)
HTF: 4h and 12h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete
Stoploss: 2.5 * ATR(14) trailing
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(len(close))
    minus_dm = np.zeros(len(close))
    
    for i in range(1, len(close)):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * (plus_dm_s / tr_s)
    minus_di = 100 * (minus_dm_s / tr_s)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
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
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    adx_12h_14 = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h_14)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volume moving average for filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15  # For weak trends
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(adx_12h_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_ma_20[i]):
            continue
        
        # === 4H TREND BIAS (MAJOR DIRECTION) ===
        # Price above 4h HMA = bullish bias (prefer longs)
        # Price below 4h HMA = bearish bias (prefer shorts)
        price_above_4h_hma = close[i] > hma_4h_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_aligned[i]
        
        # === 12H ADX REGIME FILTER ===
        # ADX > 20 = trending market (full size)
        # ADX <= 20 = ranging market (reduced size)
        trend_strong = adx_12h_aligned[i] > 20
        
        # === VOLUME FILTER ===
        # Volume > 0.6x 20-bar average (lenient to ensure trades)
        volume_ok = volume[i] > 0.6 * vol_ma_20[i]
        
        # === RSI PULLBACK ENTRY ===
        # Long: RSI pulls back to 40-50 in bullish trend
        # Short: RSI rallies to 50-60 in bearish trend
        # These are PERMISSIVE thresholds to ensure trade generation
        rsi_pullback_long = 38 <= rsi_14[i] <= 52
        rsi_pullback_short = 48 <= rsi_14[i] <= 62
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi_14[i] > 45
        rsi_momentum_short = rsi_14[i] < 55
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE if trend_strong else REDUCED_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Require: 4h bullish + RSI pullback + volume OK
        if price_above_4h_hma:
            if rsi_pullback_long and volume_ok:
                # Strong entry with momentum
                if rsi_momentum_long and trend_strong:
                    new_signal = current_size
                # Entry in weaker trend
                elif rsi_14[i] > 42:
                    new_signal = current_size * 0.8
        
        # SHORT ENTRIES
        # Require: 4h bearish + RSI pullback + volume OK
        if price_below_4h_hma:
            if rsi_pullback_short and volume_ok:
                # Strong entry with momentum
                if rsi_momentum_short and trend_strong:
                    new_signal = -current_size
                # Entry in weaker trend
                elif rsi_14[i] < 58:
                    new_signal = -current_size * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 200 bars (~8 days on 1h), allow weaker entry
        # This ensures we generate minimum trades
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if price_above_4h_hma and rsi_14[i] > 40 and rsi_14[i] < 55:
                new_signal = REDUCED_SIZE * 0.7
            elif price_below_4h_hma and rsi_14[i] > 45 and rsi_14[i] < 60:
                new_signal = -REDUCED_SIZE * 0.7
        
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
            # Exit long if 4h trend reverses bearish
            if position_side > 0 and price_below_4h_hma:
                trend_reversal = True
            # Exit short if 4h trend reverses bullish
            if position_side < 0 and price_above_4h_hma:
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