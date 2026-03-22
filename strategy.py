#!/usr/bin/env python3
"""
Experiment #027: 1d Primary + 1w HTF — Regime-Adaptive Breakout/Mean-Reversion

Hypothesis: Daily timeframe with weekly trend bias provides optimal trade frequency
(15-40 trades/year) while avoiding fee drag. Key innovations:

1. 1w HMA(21) for MAJOR trend bias (only trade WITH weekly trend)
2. Donchian(20) breakout for entry signal (proven in crypto)
3. RSI(14) filter to avoid overextended entries (30-70 range)
4. Choppiness Index(14) regime detection:
   - CHOP > 55 = range → mean reversion at Donchian bounds
   - CHOP < 45 = trend → breakout continuation
5. ATR(14) trailing stoploss at 2.5x
6. Volume confirmation (>0.7x 20-bar avg)
7. Discrete sizing: 0.25 base, 0.30 for strong confluence

Why this should work:
- 1d timeframe = fewer trades, less fee drag (target 20-50/year)
- Weekly trend filter prevents counter-trend trades in strong moves
- Choppiness Index adapts strategy to market regime
- Donchian breakout catches sustained moves in crypto
- Looser RSI filter (30-70 vs 20-80) ensures trades actually happen
- ATR stoploss protects against 2022-style crashes

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 15-40/year per symbol (60-160 total on train)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_regime_1w_hma_v1"
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
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
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
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr1_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    atr_period = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    hh_ll = high_s.rolling(window=period, min_periods=period).max().values - low_s.rolling(window=period, min_periods=period).min().values
    
    chop = 100 * (atr1_sum / atr_period) / np.maximum(hh_ll, 1e-10) * np.log10(period)
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
    """Calculate Donchian Channel upper and lower bounds."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2
    
    return upper, lower, mid

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
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian_channels(high, low, 20)
    
    # Volume SMA for filter
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(chop[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(volume_sma[i]) or volume_sma[i] == 0:
            continue
        
        # === WEEKLY TREND BIAS (MAJOR) ===
        # Price above 1w HMA = bullish bias (prefer longs)
        # Price below 1w HMA = bearish bias (prefer shorts)
        trend_1w_bullish = close[i] > hma_1w_21_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range (mean reversion at Donchian bounds)
        # CHOP < 45 = trend (breakout continuation)
        # 45-55 = neutral (use breakout logic)
        is_range = chop[i] > 55
        is_trend = chop[i] < 45
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.7 * volume_sma[i]
        
        # === RSI FILTER (avoid overextended) ===
        # For longs: RSI < 70 (not overbought)
        # For shorts: RSI > 30 (not oversold)
        rsi_ok_long = rsi_14[i] < 70
        rsi_ok_short = rsi_14[i] > 30
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Long: price breaks above Donchian upper
        # Short: price breaks below Donchian lower
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === MEAN REVERSION SIGNALS (in range regime) ===
        # Long: price near Donchian lower + oversold RSI
        # Short: price near Donchian upper + overbought RSI
        mr_long = (close[i] < donchian_lower[i] * 1.02) and (rsi_14[i] < 40)
        mr_short = (close[i] > donchian_upper[i] * 0.98) and (rsi_14[i] > 60)
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Strong confluence = higher size
        strong_confluence_long = trend_1w_bullish and volume_ok and rsi_ok_long
        strong_confluence_short = trend_1w_bearish and volume_ok and rsi_ok_short
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if is_trend:
            # Trend regime: breakout with weekly trend
            if breakout_long and trend_1w_bullish and volume_ok and rsi_ok_long:
                current_size = STRONG_SIZE if strong_confluence_long else BASE_SIZE
                new_signal = current_size
        elif is_range:
            # Range regime: mean reversion at lower bound
            if mr_long and volume_ok:
                current_size = BASE_SIZE
                new_signal = current_size
        else:
            # Neutral regime: breakout with weekly trend confirmation
            if breakout_long and trend_1w_bullish and volume_ok and rsi_ok_long:
                new_signal = BASE_SIZE
        
        # SHORT ENTRIES
        if is_trend:
            # Trend regime: breakout with weekly trend
            if breakout_short and trend_1w_bearish and volume_ok and rsi_ok_short:
                current_size = STRONG_SIZE if strong_confluence_short else BASE_SIZE
                new_signal = -current_size
        elif is_range:
            # Range regime: mean reversion at upper bound
            if mr_short and volume_ok:
                current_size = BASE_SIZE
                new_signal = -current_size
        else:
            # Neutral regime: breakout with weekly trend confirmation
            if breakout_short and trend_1w_bearish and volume_ok and rsi_ok_short:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 300 bars (~10 months on 1d), allow weaker entry
        if bars_since_last_trade > 300 and new_signal == 0.0 and not in_position:
            if trend_1w_bullish and rsi_14[i] < 50 and volume_ok:
                new_signal = BASE_SIZE * 0.5
            elif trend_1w_bearish and rsi_14[i] > 50 and volume_ok:
                new_signal = -BASE_SIZE * 0.5
        
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
            if position_side > 0 and trend_1w_bearish and rsi_14[i] > 70:
                trend_reversal = True
            if position_side < 0 and trend_1w_bullish and rsi_14[i] < 30:
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