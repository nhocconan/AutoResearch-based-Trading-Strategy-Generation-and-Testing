#!/usr/bin/env python3
"""
Experiment #090: 1h Primary + 4h/12h HTF — Fisher Transform + Choppiness Regime + Volume Session

Hypothesis: Previous 1h strategies failed because they used RSI (laggy in bear markets) 
and didn't have strict enough confluence filters. Fisher Transform catches reversals 
faster than RSI, especially in bear market rallies. Combined with:
1. 4h/12h HMA trend direction (HTF bias)
2. Choppiness Index regime filter (range vs trend)
3. Volume confirmation (>0.8x 20-bar avg)
4. Session filter (8-20 UTC only - high liquidity hours)
5. Fisher Transform entry timing (crosses -1.5 long, +1.5 short)

This should generate 30-60 trades/year on 1h (within fee drag limits) while maintaining
positive Sharpe across BTC/ETH/SOL. Fisher Transform is proven to work in bear markets
where RSI fails.

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h + 12h via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete (conservative for 1h)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_vol_session_4h12h_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5 from below
    Short when Fisher crosses below +1.5 from above
    """
    # Calculate typical price
    typical = (high + low + close) / 3.0
    typical_s = pd.Series(typical)
    
    # Normalize price over lookback period
    highest = typical_s.rolling(window=period, min_periods=period).max().values
    lowest = typical_s.rolling(window=period, min_periods=period).min().values
    
    price_range = highest - lowest
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    # Normalize to -1 to +1 range
    normalized = 2.0 * (typical - lowest) / price_range - 1.0
    normalized = np.clip(normalized, -0.999, 0.999)  # avoid log(0)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    # Signal line (1-bar lag of fisher)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    atr_values = calculate_atr(high, low, close, period)
    
    # Rolling sum of ATR
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Choppiness calculation
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
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

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope over lookback period."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_volume_ratio(volume, period=20):
    """Calculate current volume vs rolling average volume."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0, posinf=1.0, neginf=1.0)
    return vol_ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, 50)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    # Calculate 12h indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    rsi_14 = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # 1h HMA for local trend
    hma_1h_21 = calculate_hma(close, 21)
    hma_1h_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    # Conservative for 1h timeframe to reduce fee drag
    BASE_SIZE = 0.25
    
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
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            continue
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        if np.isnan(hma_1h_21[i]) or np.isnan(hma_1h_50[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only for entries) ===
        current_hour = get_utc_hour(open_time[i])
        is_session_active = 8 <= current_hour <= 20
        
        # === VOLUME CONFIRMATION ===
        # Volume must be >= 0.8x average to confirm move
        volume_confirmed = vol_ratio[i] >= 0.8
        
        # === 12H TREND BIAS (MAJOR) ===
        # HMA slope > 0 = bullish bias (prefer longs)
        # HMA slope < 0 = bearish bias (prefer shorts)
        trend_12h_bullish = hma_12h_slope_aligned[i] > 0.3
        trend_12h_bearish = hma_12h_slope_aligned[i] < -0.3
        
        # Price vs 12h HMA for additional confirmation
        price_above_12h_hma = close[i] > hma_12h_21_aligned[i]
        price_below_12h_hma = close[i] < hma_12h_21_aligned[i]
        
        # === 4H TREND CONFIRMATION ===
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.5
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.5
        hma_4h_golden = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_death = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === 1H LOCAL TREND ===
        hma_1h_bullish = hma_1h_21[i] > hma_1h_50[i]
        hma_1h_bearish = hma_1h_21[i] < hma_1h_50[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        # CHOP > 55 = range market (mean revert strategy)
        # CHOP < 45 = trend market (trend follow strategy)
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.5 from below = long signal
        # Fisher crosses below +1.5 from above = short signal
        fisher_cross_long = (fisher_signal[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_cross_short = (fisher_signal[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Fisher extreme levels for mean reversion
        fisher_deep_oversold = fisher[i] < -2.0
        fisher_deep_overbought = fisher[i] > 2.0
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in transitional markets or weak volume
        if not is_range_market and not is_trend_market:
            current_size = BASE_SIZE * 0.6
        if not volume_confirmed:
            current_size = current_size * 0.7
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # Minimum bars between trades to reduce churn (1h = 24 bars = 1 day)
        min_bars_between = 24
        
        # LONG ENTRIES - require 3+ confluence
        if is_range_market:
            # Mean reversion in range: Fisher deep oversold + RSI confirmation
            if (fisher_deep_oversold or fisher_cross_long) and rsi_oversold:
                # Need HTF bias or price above 12h HMA
                if trend_12h_bullish or price_above_12h_hma:
                    if is_session_active and volume_confirmed and bars_since_last_trade >= min_bars_between:
                        new_signal = current_size
        elif is_trend_market:
            # Trend following: Fisher pullback entry in uptrend
            if trend_12h_bullish and trend_4h_bullish:
                # Fisher crosses up from oversold in uptrend
                if fisher_cross_long and hma_1h_bullish:
                    if is_session_active and volume_confirmed and bars_since_last_trade >= min_bars_between:
                        new_signal = current_size
                # Or Fisher not extreme but trend strong
                elif fisher[i] < 0 and hma_1h_bullish and rsi_14[i] < 50:
                    if is_session_active and volume_confirmed and bars_since_last_trade >= min_bars_between:
                        new_signal = current_size * 0.8
        else:
            # Transitional: only strong signals with HTF confirmation
            if trend_12h_bullish and fisher_cross_long:
                if is_session_active and volume_confirmed and bars_since_last_trade >= min_bars_between:
                    new_signal = current_size * 0.5
        
        # SHORT ENTRIES - require 3+ confluence
        if is_range_market:
            # Mean reversion in range: Fisher deep overbought + RSI confirmation
            if (fisher_deep_overbought or fisher_cross_short) and rsi_overbought:
                # Need HTF bias or price below 12h HMA
                if trend_12h_bearish or price_below_12h_hma:
                    if is_session_active and volume_confirmed and bars_since_last_trade >= min_bars_between:
                        new_signal = -current_size
        elif is_trend_market:
            # Trend following: Fisher pullback entry in downtrend
            if trend_12h_bearish and trend_4h_bearish:
                # Fisher crosses down from overbought in downtrend
                if fisher_cross_short and hma_1h_bearish:
                    if is_session_active and volume_confirmed and bars_since_last_trade >= min_bars_between:
                        new_signal = -current_size
                # Or Fisher not extreme but trend strong
                elif fisher[i] > 0 and hma_1h_bearish and rsi_14[i] > 50:
                    if is_session_active and volume_confirmed and bars_since_last_trade >= min_bars_between:
                        new_signal = -current_size * 0.8
        else:
            # Transitional: only strong signals with HTF confirmation
            if trend_12h_bearish and fisher_cross_short:
                if is_session_active and volume_confirmed and bars_since_last_trade >= min_bars_between:
                    new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 200 bars (~8 days on 1h), allow weaker entry
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if trend_12h_bullish and fisher[i] < -1.0 and rsi_14[i] < 40:
                if volume_confirmed:
                    new_signal = current_size * 0.4
            elif trend_12h_bearish and fisher[i] > 1.0 and rsi_14[i] > 60:
                if volume_confirmed:
                    new_signal = -current_size * 0.4
        
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
        
        # === REGIME REVERSAL EXIT ===
        # Exit if HTF trend changes against position
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if 12h trend becomes strongly bearish
            if position_side > 0 and trend_12h_bearish and hma_12h_slope_aligned[i] < -0.5:
                regime_reversal = True
            # Exit short if 12h trend becomes strongly bullish
            if position_side < 0 and trend_12h_bullish and hma_12h_slope_aligned[i] > 0.5:
                regime_reversal = True
        
        # Apply stoploss or regime reversal
        if stoploss_triggered or regime_reversal:
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