#!/usr/bin/env python3
"""
Experiment #003: 1d Primary + 1w HTF — Donchian Breakout with Regime Filter

Hypothesis: Daily timeframe with weekly trend filter reduces whipsaws while
capturing major moves. Donchian breakouts work well on 1d for crypto.

Key components:
1. 1w HMA(21) for MAJOR trend bias (only trade WITH weekly trend)
2. 1d Donchian(20) breakout for entry signals
3. Choppiness Index(14) to avoid breakout failures in choppy markets
4. ATR(14) trailing stoploss at 2.5x
5. Volume confirmation (>0.8x 20-day avg)
6. Asymmetric sizing: 0.30 with trend, 0.20 against

Why this should work on 1d:
- Weekly filter prevents counter-trend trades that fail in 2022 crash
- Donchian breakout captures sustained moves (20-50 trades/year target)
- Choppiness filter avoids false breakouts in range markets
- 1d timeframe = natural trade frequency (no fee drag from overtrading)
- Discrete sizing minimizes churn

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 20-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_chop_hma_1w_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index = 100 * (ATR(1) sum / ATR(period)) / (Highest High - Lowest Low) * log10(period)
    
    CHOP > 61.8 = range/choppy
    CHOP < 38.2 = trending
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr1_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    atr_period = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    hh_ll = high_s.rolling(window=period, min_periods=period).max().values - low_s.rolling(window=period, min_periods=period).min().values
    
    chop = 100 * (atr1_sum / atr_period) / hh_ll * np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    
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

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian Channel upper and lower bands."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    
    return upper, lower, middle

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop = calculate_choppiness_index(high, low, close, 14)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian_channels(high, low, 20)
    
    # Volume SMA for filter
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Price momentum (ROC)
    close_s = pd.Series(close)
    roc_10 = close_s.pct_change(periods=10).values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE_WITH_TREND = 0.30
    BASE_SIZE_COUNTER = 0.20
    
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
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(chop[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(volume_sma[i]) or volume_sma[i] == 0:
            continue
        
        # === WEEKLY TREND BIAS (MAJOR) ===
        # Price above 1w HMA = bullish bias (prefer longs)
        # Price below 1w HMA = bearish bias (prefer shorts)
        # Also check HMA slope for confirmation
        hma_slope_bullish = hma_1w_21_aligned[i] > hma_1w_21_aligned[i-7] if i >= 107 else False
        hma_slope_bearish = hma_1w_21_aligned[i] < hma_1w_21_aligned[i-7] if i >= 107 else False
        
        trend_1w_bullish = close[i] > hma_1w_21_aligned[i] and hma_slope_bullish
        trend_1w_bearish = close[i] < hma_1w_21_aligned[i] and hma_slope_bearish
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range (breakouts likely to fail)
        # CHOP < 45 = trending (breakouts likely to succeed)
        is_trending = chop[i] < 45
        is_choppy = chop[i] > 55
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * volume_sma[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Breakout above upper band = long signal
        # Breakout below lower band = short signal
        breakout_long = close[i] > donchian_upper[i-1]  # Break above previous upper
        breakout_short = close[i] < donchian_lower[i-1]  # Break below previous lower
        
        # === RSI CONFIRMATION ===
        # For longs: RSI > 50 (momentum confirmation)
        # For shorts: RSI < 50 (momentum confirmation)
        rsi_confirms_long = rsi_14[i] > 50
        rsi_confirms_short = rsi_14[i] < 50
        
        # === POSITION SIZING ===
        # Full size with trend, reduced size counter-trend
        current_size_with_trend = BASE_SIZE_WITH_TREND
        current_size_counter = BASE_SIZE_COUNTER
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Primary: Trending regime + weekly bullish + Donchian breakout + volume + RSI
        # Secondary: After long dormancy, allow weaker entry
        if is_trending and trend_1w_bullish and breakout_long and volume_ok and rsi_confirms_long:
            new_signal = current_size_with_trend
        elif is_choppy and trend_1w_bullish and breakout_long and rsi_confirms_long:
            # In choppy market, only enter with strong weekly trend
            new_signal = current_size_counter * 0.8
        
        # SHORT ENTRIES
        # Primary: Trending regime + weekly bearish + Donchian breakout + volume + RSI
        # Secondary: After long dormancy, allow weaker entry
        if is_trending and trend_1w_bearish and breakout_short and volume_ok and rsi_confirms_short:
            new_signal = -current_size_with_trend
        elif is_choppy and trend_1w_bearish and breakout_short and rsi_confirms_short:
            # In choppy market, only enter with strong weekly trend
            new_signal = -current_size_counter * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 300 bars (~10 months on 1d), allow weaker entry
        if bars_since_last_trade > 300 and new_signal == 0.0 and not in_position:
            if trend_1w_bullish and breakout_long:
                new_signal = current_size_counter * 0.7
            elif trend_1w_bearish and breakout_short:
                new_signal = -current_size_counter * 0.7
        
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
            # Exit long if weekly trend turns bearish + RSI overbought
            if position_side > 0 and trend_1w_bearish and rsi_14[i] > 70:
                trend_reversal = True
            # Exit short if weekly trend turns bullish + RSI oversold
            if position_side < 0 and trend_1w_bullish and rsi_14[i] < 30:
                trend_reversal = True
        
        # === CHOPPINESS EXIT ===
        # If market becomes very choppy while in position, reduce exposure
        choppy_exit = False
        if in_position and position_side != 0 and chop[i] > 65:
            # Market became very choppy, exit to avoid whipsaw
            choppy_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or choppy_exit:
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
                # Flip position
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