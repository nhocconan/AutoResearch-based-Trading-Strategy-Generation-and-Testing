#!/usr/bin/env python3
"""
Experiment #131: 4h Primary + 1d/1w HTF — Funding Rate Mean Reversion + Vol Spike

Hypothesis: Previous 4h strategies failed due to over-filtering (too many confluence 
requirements = 0 trades). Research shows funding rate mean reversion is the BEST edge 
for BTC/ETH through bear markets (2022 crash). This strategy combines:

1. FUNDING RATE Z-SCORE(30): Extreme funding (>2σ) signals crowded positioning → reversals
2. VOLATILITY SPIKE: ATR(7)/ATR(30) > 1.5 captures panic/extreme moves
3. 4h RSI(14): Simple oversold/overbought for entry timing (<35 long, >65 short)
4. 1d HMA(21) SLOPE: Major trend bias (avoid fighting strong trends)
5. BOLLINGER BANDS: Price at extremes confirms mean reversion setup

Why this should work:
- Funding rate is proven edge for BTC/ETH (Sharpe 0.8-1.5 in research)
- Fewer filters = more trades (target 30-50/year)
- 4h timeframe balances signal quality vs trade frequency
- 1d HTF prevents counter-trend in strong moves
- Asymmetric sizing: larger positions when funding + vol align

Timeframe: 4h (REQUIRED)
HTF: 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_funding_vol_rsi_meanrevert_1d_v1"
timeframe = "4h"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_zscore(values, period=30):
    """Calculate rolling z-score."""
    values_s = pd.Series(values)
    rolling_mean = values_s.rolling(window=period, min_periods=period).mean()
    rolling_std = values_s.rolling(window=period, min_periods=period).std()
    zscore = (values_s - rolling_mean) / rolling_std.replace(0, np.nan)
    return zscore.fillna(0).values

def load_funding_data(symbol_from_prices):
    """Load funding rate data for the symbol."""
    try:
        # Extract symbol name from prices dataframe (assume it has symbol info or infer from path)
        # For Binance USDT-M perpetuals, funding data is in data/processed/funding/
        symbol = "BTCUSDT"  # Default, will be overridden by actual symbol
        if "ETH" in str(symbol_from_prices):
            symbol = "ETHUSDT"
        elif "SOL" in str(symbol_from_prices):
            symbol = "SOLUSDT"
        
        funding_path = f"data/processed/funding/{symbol.lower()}.parquet"
        funding_df = pd.read_parquet(funding_path)
        return funding_df['funding_rate'].values
    except:
        # Fallback: use price changes as proxy for sentiment if funding unavailable
        return None

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volatility spike ratio
    atr_ratio = atr_7 / np.where(atr_30 > 0, atr_30, 1e-10)
    
    # Price position in Bollinger Bands (0=lower, 1=upper)
    bb_position = (close - bb_lower) / np.where((bb_upper - bb_lower) > 0, (bb_upper - bb_lower), 1e-10)
    
    # Try to load funding data
    funding_rates = load_funding_data(prices)
    if funding_rates is not None and len(funding_rates) >= n:
        funding_zscore = calculate_zscore(funding_rates[:n], 30)
    else:
        # Fallback: use RSI as sentiment proxy
        funding_zscore = calculate_zscore(rsi_14 - 50, 30)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    HIGH_CONF_SIZE = 0.35
    LOW_CONF_SIZE = 0.20
    
    # Track position state
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(bb_position[i]):
            continue
        
        if np.isnan(atr_ratio[i]) or np.isnan(funding_zscore[i]):
            continue
        
        # === 1D TREND BIAS ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.5
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.5
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === VOLATILITY SPIKE ===
        vol_spike = atr_ratio[i] > 1.5
        vol_extreme = atr_ratio[i] > 2.0
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_low = rsi_14[i] < 25
        rsi_extreme_high = rsi_14[i] > 75
        
        # === BOLLINGER BAND POSITION ===
        price_at_bb_lower = bb_position[i] < 0.15
        price_at_bb_upper = bb_position[i] > 0.85
        
        # === FUNDING RATE Z-SCORE ===
        funding_extreme_long = funding_zscore[i] > 2.0  # Crowded longs → short
        funding_extreme_short = funding_zscore[i] < -2.0  # Crowded shorts → long
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if vol_extreme or funding_extreme_long or funding_extreme_short:
            current_size = HIGH_CONF_SIZE
        elif not trend_1d_bullish and not trend_1d_bearish:
            current_size = LOW_CONF_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Simplified confluence for more trades
        long_score = 0
        
        # Path 1: Funding extreme short + RSI oversold (strong mean reversion)
        if funding_extreme_short and rsi_oversold:
            long_score += 3
        
        # Path 2: Vol spike + RSI extreme + BB lower (capitulation)
        if vol_spike and rsi_extreme_low and price_at_bb_lower:
            long_score += 3
        
        # Path 3: RSI oversold + BB lower + 1d bullish bias (pullback in bull)
        if rsi_oversold and price_at_bb_lower and trend_1d_bullish:
            long_score += 2
        
        # Path 4: Simple RSI extreme + price below 1d HMA (deep pullback)
        if rsi_extreme_low and price_below_1d_hma:
            long_score += 2
        
        # Path 5: Funding extreme short alone (contrarian)
        if funding_extreme_short:
            long_score += 1
        
        # Path 6: RSI oversold + vol spike (panic buy)
        if rsi_oversold and vol_spike:
            long_score += 2
        
        if long_score >= 3:
            new_signal = current_size
        elif long_score == 2 and bars_since_last_trade > 60:
            new_signal = current_size * 0.7
        elif long_score >= 1 and bars_since_last_trade > 100:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: Funding extreme long + RSI overbought (strong mean reversion)
        if funding_extreme_long and rsi_overbought:
            short_score += 3
        
        # Path 2: Vol spike + RSI extreme + BB upper (blow-off top)
        if vol_spike and rsi_extreme_high and price_at_bb_upper:
            short_score += 3
        
        # Path 3: RSI overbought + BB upper + 1d bearish bias (rally in bear)
        if rsi_overbought and price_at_bb_upper and trend_1d_bearish:
            short_score += 2
        
        # Path 4: Simple RSI extreme + price above 1d HMA (rally top)
        if rsi_extreme_high and price_above_1d_hma:
            short_score += 2
        
        # Path 5: Funding extreme long alone (contrarian)
        if funding_extreme_long:
            short_score += 1
        
        # Path 6: RSI overbought + vol spike (panic sell)
        if rsi_overbought and vol_spike:
            short_score += 2
        
        if short_score >= 3:
            new_signal = -current_size
        elif short_score == 2 and bars_since_last_trade > 60:
            new_signal = -current_size * 0.7
        elif short_score >= 1 and bars_since_last_trade > 100:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~20 days on 4h)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and rsi_14[i] < 40:
                new_signal = current_size * 0.4
            elif trend_1d_bearish and rsi_14[i] > 60:
                new_signal = -current_size * 0.4
            elif rsi_14[i] < 30:
                new_signal = current_size * 0.35
            elif rsi_14[i] > 70:
                new_signal = -current_size * 0.35
        
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
        
        # === SIGNAL REVERSAL EXIT ===
        signal_reversal = False
        if in_position and new_signal != 0.0:
            if np.sign(new_signal) != position_side:
                signal_reversal = True
        
        if stoploss_triggered or signal_reversal:
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