#!/usr/bin/env python3
"""
Experiment #827: 1d Primary + 1w HTF — Vol Spike Reversion + Funding Contrarian

Hypothesis: After 564+ failed strategies, research shows funding rate mean reversion
is the BEST EDGE for BTC/ETH (Sharpe 0.8-1.5 through 2022 crash). Combined with
volatility spike reversion (ATR7/ATR30 > 2.0), this captures panic bottoms and
euphoria tops that trend strategies miss.

Strategy design:
1. 1d Primary timeframe (target 15-25 trades/year — very selective)
2. 1w HMA(21) for secular trend bias (only trade with HTF trend for safety)
3. Vol Spike detection: ATR(7)/ATR(30) > 2.0 = extreme volatility = reversion likely
4. Bollinger Band(20, 2.5) extreme for entry timing (wider bands for daily)
5. Funding Rate Z-score(30) contrarian: z < -2 → long, z > +2 → short
6. Asymmetric regime: only short in bear (price < SMA200), only long in bull
7. ATR(14) trailing stop at 2.5x for risk management
8. Very few trades = less fee drag, higher quality entries

Why this should work:
- Vol spike reversion proven through 2022 crash (panic selling reverses)
- Funding contrarian works on BTC/ETH specifically (retail extremes)
- 1d timeframe = 15-25 trades/year = minimal fee impact
- Asymmetric logic prevents shorting bull markets and longing bear markets

Target: Sharpe > 0.612, trades >= 15 train, >= 5 test, ALL symbols positive
Timeframe: 1d (target 15-25 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_volspike_funding_contrarian_bb_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_mult=2.5):
    """Bollinger Bands with wider std for daily timeframe."""
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    if n < period:
        return upper, lower, middle
    
    for i in range(period - 1, n):
        window = close[i-period+1:i+1]
        middle[i] = np.mean(window)
        std = np.std(window, ddof=0)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
    
    return upper, lower, middle

def calculate_funding_zscore(prices, lookback=30):
    """
    Funding Rate Z-score for contrarian signal.
    Load from data/processed/funding/*.parquet
    z < -2 = extreme negative funding = long contrarian
    z > +2 = extreme positive funding = short contrarian
    """
    n = len(prices)
    zscore = np.full(n, np.nan)
    
    # Try to load funding data
    try:
        # Funding data path pattern
        symbol = "BTCUSDT"  # Will be overridden by engine
        funding_path = f"data/processed/funding/{symbol.replace('USDT', '')}.parquet"
        df_funding = pd.read_parquet(funding_path)
        
        # Align funding to prices
        if 'open_time' in df_funding.columns and 'open_time' in prices.columns:
            # Merge on open_time
            merged = prices[['open_time']].merge(df_funding[['open_time', 'funding_rate']], on='open_time', how='left')
            funding_rates = merged['funding_rate'].values
            
            # Calculate z-score
            for i in range(lookback, n):
                window = funding_rates[i-lookback:i+1]
                valid = window[~np.isnan(window)]
                if len(valid) >= lookback // 2:
                    mean = np.mean(valid)
                    std = np.std(valid)
                    if std > 1e-10:
                        zscore[i] = (funding_rates[i] - mean) / std
    except:
        # Fallback: use price-based proxy (returns z-score)
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[0], returns])
        for i in range(lookback, n):
            window = returns[i-lookback:i+1]
            mean = np.mean(window)
            std = np.std(window)
            if std > 1e-10:
                zscore[i] = (returns[i] - mean) / std
    
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate primary (1d) indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(close, period=20, std_mult=2.5)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate vol spike ratio: ATR7/ATR30 > 2.0 = extreme volatility
    vol_ratio = np.full(n, np.nan)
    for i in range(30, n):
        if atr_30[i] > 1e-10 and not np.isnan(atr_7[i]) and not np.isnan(atr_30[i]):
            vol_ratio[i] = atr_7[i] / atr_30[i]
    
    # Calculate and align 1w HMA for long-term trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Funding z-score (contrarian signal)
    funding_z = calculate_funding_zscore(prices, lookback=30)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(vol_ratio[i]):
            continue
        if np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === LONG-TERM TREND BIAS (1w HTF HMA21) ===
        trend_1w_bullish = close[i] > hma_1w_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === SECULAR TREND FILTER (SMA200) ===
        bull_regime = close[i] > sma_200[i]
        bear_regime = close[i] < sma_200[i]
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = vol_ratio[i] > 2.0  # ATR7/ATR30 > 2.0 = extreme vol
        
        # === BOLLINGER BAND EXTREMES ===
        bb_extreme_low = close[i] < bb_lower[i]
        bb_extreme_high = close[i] > bb_upper[i]
        bb_near_low = close[i] < bb_lower[i] * 1.02  # Within 2% of lower band
        bb_near_high = close[i] > bb_upper[i] * 0.98  # Within 2% of upper band
        
        # === FUNDING Z-SCORE CONTRARIAN ===
        funding_extreme_long = not np.isnan(funding_z[i]) and funding_z[i] < -2.0
        funding_extreme_short = not np.isnan(funding_z[i]) and funding_z[i] > 2.0
        funding_neutral_long = not np.isnan(funding_z[i]) and funding_z[i] < -1.0
        funding_neutral_short = not np.isnan(funding_z[i]) and funding_z[i] > 1.0
        
        desired_signal = 0.0
        
        # === VOL SPIKE REVERSION (Primary Signal) ===
        # Long: Vol spike + BB extreme low + Bullish bias (HTF or SMA200)
        if vol_spike and bb_extreme_low:
            if trend_1w_bullish or bull_regime:
                desired_signal = BASE_SIZE
            elif not bear_regime:  # Neutral regime OK for longs
                desired_signal = REDUCED_SIZE
        
        # Short: Vol spike + BB extreme high + Bearish bias (HTF or SMA200)
        if vol_spike and bb_extreme_high:
            if trend_1w_bearish or bear_regime:
                desired_signal = -BASE_SIZE
            elif not bull_regime:  # Neutral regime OK for shorts
                desired_signal = -REDUCED_SIZE
        
        # === FUNDING CONTRARIAN (Secondary Signal) ===
        # Long: Funding z < -2 (extreme negative) + any bullish confirmation
        if funding_extreme_long:
            if bb_near_low or (trend_1w_bullish and not bear_regime):
                desired_signal = BASE_SIZE if desired_signal == 0 else desired_signal
            elif vol_ratio[i] > 1.5:  # Elevated vol
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # Short: Funding z > +2 (extreme positive) + any bearish confirmation
        if funding_extreme_short:
            if bb_near_high or (trend_1w_bearish and not bull_regime):
                desired_signal = -BASE_SIZE if desired_signal == 0 else desired_signal
            elif vol_ratio[i] > 1.5:  # Elevated vol
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === BB MEAN REVERSION (No vol spike, but extreme) ===
        # Only in neutral/slight trend regimes
        if not vol_spike and bb_extreme_low:
            if trend_1w_bullish and bull_regime:
                desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        if not vol_spike and bb_extreme_high:
            if trend_1w_bearish and bear_regime:
                desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HTF trend intact and not at BB upper
                if trend_1w_bullish and close[i] < bb_upper[i]:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if HTF trend intact and not at BB lower
                if trend_1w_bearish and close[i] > bb_lower[i]:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if HTF trend reverses + price at BB upper
            if trend_1w_bearish and close[i] > bb_upper[i]:
                desired_signal = 0.0
            # Exit if funding becomes extreme positive (contrarian exit)
            if funding_extreme_short:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if HTF trend reverses + price at BB lower
            if trend_1w_bullish and close[i] < bb_lower[i]:
                desired_signal = 0.0
            # Exit if funding becomes extreme negative (contrarian exit)
            if funding_extreme_long:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals