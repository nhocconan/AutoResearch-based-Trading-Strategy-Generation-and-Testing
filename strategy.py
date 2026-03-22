#!/usr/bin/env python3
"""
Experiment #008: 30m Trend Pullback with 4h/1d HTF Filter

Hypothesis: Previous mean-reversion strategies (CRSI, CHOP) failed because crypto
trends persist longer than expected. Research shows trend-following with pullback
entries works better for BTC/ETH than pure mean reversion. This strategy uses:

1. 1d HMA(21) for MAJOR trend bias - only trade in direction of daily trend
2. 4h HMA(21) for INTERMEDIATE trend confirmation - need alignment with daily
3. 30m EMA(21) pullback entry - enter when price retraces to EMA within HTF trend
4. Volume confirmation - volume > 1.2x 20-bar average (institutional interest)
5. Session filter - only trade 8-20 UTC (high liquidity, avoid Asia overnight)
6. ATR(14) trailing stop - 2.5x ATR from entry/extreme
7. RSI(14) momentum filter - RSI > 45 for longs, < 55 for shorts (avoid weak momentum)

Why this should work:
- HTF (1d/4h) determines DIRECTION, 30m only for ENTRY TIMING
- Pullback entries have better R:R than breakouts (enter at support/resistance)
- Volume filter avoids false breakouts during low liquidity
- Session filter reduces whipsaw during Asia session (low volume)
- Conservative sizing (0.25) protects against 2022-style crashes
- Target: 40-70 trades/year on 30m (strict confluence = fewer trades)

Timeframe: 30m (REQUIRED for Experiment #008)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 base, 0.15 reduced (discrete levels)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_pullback_hma_4h_1d_volume_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) period
    """
    close_s = pd.Series(close)
    n = period
    
    def wma(series, span):
        return series.ewm(span=span, min_periods=span, adjust=False).mean()
    
    half = int(n / 2)
    sqrt_n = int(np.sqrt(n))
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_ema(close, period=21):
    """Calculate EMA using standard formula."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume for volume confirmation."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean()
    return vol_sma.values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    ts_seconds = open_time / 1000
    utc_hour = pd.to_datetime(ts_seconds, unit='s').hour
    return utc_hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for major trend
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h HMA for intermediate trend
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 30m indicators
    ema_30m_21 = calculate_ema(close, period=21)
    ema_30m_50 = calculate_ema(close, period=50)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, 14)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_4h_21_aligned[i]):
            continue
        
        if np.isnan(ema_30m_21[i]) or np.isnan(ema_30m_50[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_sma_20[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === 1D MAJOR TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H INTERMEDIATE TREND ===
        hma_4h_bullish = close[i] > hma_4h_21_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === 30M SHORT-TERM TREND ===
        ema_bullish = ema_30m_21[i] > ema_30m_50[i]
        ema_bearish = ema_30m_21[i] < ema_30m_50[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.2 * vol_sma_20[i]
        
        # === RSI MOMENTUM FILTER ===
        rsi_long_ok = rsi_14[i] > 45  # Not oversold (momentum present)
        rsi_short_ok = rsi_14[i] < 55  # Not overbought (momentum present)
        
        # === PULLBACK ENTRY LOGIC ===
        # Long: price pulls back to EMA21 but stays above it, within HTF uptrend
        price_near_ema_long = (close[i] >= ema_30m_21[i] * 0.995) and (close[i] <= ema_30m_21[i] * 1.015)
        price_above_ema_long = close[i] > ema_30m_21[i]
        
        # Short: price pulls back to EMA21 but stays below it, within HTF downtrend
        price_near_ema_short = (close[i] <= ema_30m_21[i] * 1.005) and (close[i] >= ema_30m_21[i] * 0.985)
        price_below_ema_short = close[i] < ema_30m_21[i]
        
        # === ENTRY SCORING ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: All HTF bullish + pullback to EMA + volume + session + RSI
        long_score = 0
        
        # HTF trend alignment (need both 1d and 4h bullish)
        if daily_bullish and hma_4h_bullish:
            long_score += 3.0
        elif daily_bullish or hma_4h_bullish:
            long_score += 1.0
        
        # 30m trend confirmation
        if ema_bullish:
            long_score += 1.0
        
        # Pullback entry (price near EMA21)
        if price_near_ema_long and price_above_ema_long:
            long_score += 2.0
        elif price_above_ema_long:
            long_score += 0.5
        
        # Volume confirmation
        if volume_confirmed:
            long_score += 1.5
        
        # Session filter
        if in_session:
            long_score += 1.0
        
        # RSI momentum
        if rsi_long_ok:
            long_score += 1.0
        
        # Enter long if score >= 7.0 (very strict confluence)
        if long_score >= 7.0:
            new_signal = BASE_SIZE
        
        # SHORT ENTRY: All HTF bearish + pullback to EMA + volume + session + RSI
        short_score = 0
        
        # HTF trend alignment (need both 1d and 4h bearish)
        if daily_bearish and hma_4h_bearish:
            short_score += 3.0
        elif daily_bearish or hma_4h_bearish:
            short_score += 1.0
        
        # 30m trend confirmation
        if ema_bearish:
            short_score += 1.0
        
        # Pullback entry (price near EMA21)
        if price_near_ema_short and price_below_ema_short:
            short_score += 2.0
        elif price_below_ema_short:
            short_score += 0.5
        
        # Volume confirmation
        if volume_confirmed:
            short_score += 1.5
        
        # Session filter
        if in_session:
            short_score += 1.0
        
        # RSI momentum
        if rsi_short_ok:
            short_score += 1.0
        
        # Enter short if score >= 7.0 (very strict confluence)
        if short_score >= 7.0:
            new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 150 bars (~3 days on 30m), allow slightly weaker entry
        if bars_since_last_trade > 150 and new_signal == 0.0 and not in_position:
            if daily_bullish and hma_4h_bullish and price_above_ema_long and in_session:
                new_signal = REDUCED_SIZE
            elif daily_bearish and hma_4h_bearish and price_below_ema_short and in_session:
                new_signal = -REDUCED_SIZE
        
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
            # Exit long if major trend turns bearish
            if position_side > 0 and daily_bearish and hma_4h_bearish:
                trend_reversal = True
            # Exit short if major trend turns bullish
            if position_side < 0 and daily_bullish and hma_4h_bullish:
                trend_reversal = True
        
        # === RSI EXTREME EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long if RSI goes overbought (>75)
            if position_side > 0 and rsi_14[i] > 75:
                rsi_exit = True
            # Exit short if RSI goes oversold (<25)
            if position_side < 0 and rsi_14[i] < 25:
                rsi_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or rsi_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals