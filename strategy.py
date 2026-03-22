#!/usr/bin/env python3
"""
Experiment #035: 1h Primary + 4h/1d HTF — ADX Regime + RSI Mean Reversion

Hypothesis: Previous Connors RSI + Choppiness strategies failed due to complexity
and over-filtering (0 trades). This strategy uses SIMPLER, more robust signals:

1. 1d SMA(200) for MAJOR trend bias (price above = long bias)
2. 4h HMA(21) for INTERMEDIATE trend confirmation
3. ADX(14) regime filter: ADX>25=trend, ADX<20=range
4. RSI(14) extremes: <35 oversold long, >65 overbought short
5. Volume filter (>0.7x 20-bar avg)
6. Session filter (8-20 UTC - high liquidity hours)
7. ATR(14) trailing stoploss at 2.5x

Why this should work:
- ADX is more reliable than Choppiness Index for regime detection
- RSI(14) is simpler and generates more signals than Connors RSI
- 1h entries within 4h/1d trend = proven pattern from baseline
- Looser RSI thresholds (35/65 vs 20/80) = more trades generated
- Session filter avoids Asian session whipsaws

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 base (discrete: 0.0, ±0.25, ±0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 40-80/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_adx_regime_rsi_4h1d_v1"
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
    """
    Calculate ADX (Average Directional Index) for trend strength.
    ADX > 25 = trending market
    ADX < 20 = ranging/choppy market
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smooth with Wilder's method
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    adx = adx.fillna(0).values
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // (1000 * 3600)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    sma_1d_200 = calculate_sma(df_1d['close'].values, 200)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    sma_1d_200_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_200)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    
    # Volume SMA for filter
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    BOOST_SIZE = 0.35
    
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
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(sma_1d_200_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        if np.isnan(volume_sma[i]) or volume_sma[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === 1D TREND BIAS (MAJOR) ===
        # Price above 1d SMA200 = bullish bias (prefer longs)
        # Price below 1d SMA200 = bearish bias (prefer shorts)
        trend_1d_bullish = close[i] > sma_1d_200_aligned[i]
        trend_1d_bearish = close[i] < sma_1d_200_aligned[i]
        
        # === 4H TREND CONFIRMATION (INTERMEDIATE) ===
        trend_4h_bullish = close[i] > hma_4h_21_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === ADX REGIME FILTER ===
        # ADX > 25 = trending (follow trend direction)
        # ADX < 20 = ranging (mean reversion)
        # ADX 20-25 = neutral (can trade either way with confirmation)
        is_trending = adx_14[i] > 25
        is_ranging = adx_14[i] < 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.7 * volume_sma[i]
        
        # === RSI EXTREMES ===
        # RSI < 35 = oversold (long opportunity)
        # RSI > 65 = overbought (short opportunity)
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # Moderate RSI for continuation
        rsi_moderate_low = rsi_14[i] < 50
        rsi_moderate_high = rsi_14[i] > 50
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Trending regime: 1d bullish + 4h bullish + RSI moderate (pullback entry)
        # Ranging regime: 4h bullish + RSI oversold (mean reversion)
        if is_trending:
            if trend_1d_bullish and trend_4h_bullish and rsi_moderate_low and volume_ok and in_session:
                new_signal = current_size
        elif is_ranging:
            if trend_4h_bullish and rsi_oversold and volume_ok and in_session:
                new_signal = current_size
        else:
            # Neutral regime: require strong confluence
            if trend_1d_bullish and trend_4h_bullish and rsi_oversold and volume_ok and in_session:
                new_signal = current_size
        
        # SHORT ENTRIES
        # Trending regime: 1d bearish + 4h bearish + RSI moderate (pullback entry)
        # Ranging regime: 4h bearish + RSI overbought (mean reversion)
        if is_trending:
            if trend_1d_bearish and trend_4h_bearish and rsi_moderate_high and volume_ok and in_session:
                new_signal = -current_size
        elif is_ranging:
            if trend_4h_bearish and rsi_overbought and volume_ok and in_session:
                new_signal = -current_size
        else:
            # Neutral regime: require strong confluence
            if trend_1d_bearish and trend_4h_bearish and rsi_overbought and volume_ok and in_session:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 150 bars (~6 days on 1h), allow weaker entry to ensure trade generation
        if bars_since_last_trade > 150 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and trend_4h_bullish and rsi_14[i] < 45 and volume_ok:
                new_signal = current_size * 0.8
            elif trend_1d_bearish and trend_4h_bearish and rsi_14[i] > 55 and volume_ok:
                new_signal = -current_size * 0.8
        
        # === BOOST SIZE ON STRONG SIGNALS ===
        # If all conditions align perfectly, increase position size
        if new_signal != 0.0:
            if is_trending and trend_1d_bullish == (new_signal > 0) and trend_4h_bullish == (new_signal > 0):
                if abs(rsi_14[i] - 50) > 20:  # RSI far from neutral
                    new_signal = np.sign(new_signal) * BOOST_SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and trend_1d_bearish and rsi_14[i] > 65:
                trend_reversal = True
            if position_side < 0 and trend_1d_bullish and rsi_14[i] < 35:
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